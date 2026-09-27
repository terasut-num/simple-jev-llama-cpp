"""Single-file GGUF classifier server using llama.cpp (llama-cpp-python).

Run directly from a checkout (the sibling common/ folder is required)::

    python hf-server/hf_server.py --model /path/to/model.gguf --device cpu

Or install with pip install -e './hf-server[test]' and run::

    simple-jev --model organization/model --device auto
    python -m hf_server --model organization/model --device auto

Request flow:
    llama.cpp: ClassifierRequest -> PromptCompiler -> LlamaCppBackend -> common.build_response
    Laya: ClassifierRequest -> LayaBackend -> native encoder scoring -> response

common owns validation, default versioned classifier wording, label semantics,
and answer math. hf_prompt_policies adds startup-selected formatting and binary Noul
adaptation without modifying common. This file owns text/image chat role assembly,
native chat/tokenizer boundaries,
shared-prefix inference, queue/cancellation controls, usage accounting, HTTP, and
startup. Model weights load only when load_service/main is called.

The inference engine is llama.cpp through llama-cpp-python. Models must be GGUF
files (or Hugging Face repositories containing one); weights are offloaded to
Vulkan-capable GPUs when --device auto requests it. The engine never samples
tokens: it scores the next-token logits of each branch's answer boundary.

The sections below follow the data flow and retain the implementation notes for
KV ownership, per-row logit selection, and asynchronous cleanup. Tests live in
tests/; no separate simple_jev package or duplicate baseline template is needed.
"""

import argparse
import asyncio
import copy
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import ValidationError

# Direct script execution puts hf-server/, not the checkout root, on sys.path.
# Prefer the sibling common source when running from this repo; an installed
# wheel instead imports its bundled common package through normal resolution.
_checkout_root = Path(__file__).resolve().parent.parent
if (_checkout_root / "common" / "prompt_builder.py").is_file():
    sys.path.insert(0, str(_checkout_root))

from common import (
    ClassifierRequest,
    PromptPlan,
    build_response,
    prepare_prompt,
)
from common.prompt_builder import DEFAULT_TEMPLATE_VERSION, canonical
from hf_prompt_policies import (
    KNOWN_PROFILES, PROFILE_FIELDS, PROMPT_POLICIES, format_branch, prepare_policy,
    restore_binary_noul, validate_policy, resolve_prompt_policy,
)

# Shared plan to native chat and tokens


@dataclass(frozen=True)
class Branch:
    """One compiled question, identified by its plan-local branch ID.

    token_ids encodes the complete prompt through the incomplete assistant answer
    prefix. output_ids contains the next-token vocabulary IDs, in the same order
    as the shared question's output_labels. Messages/prefix remain available for
    inspection; they are not reconstructed from tokens during inference.
    reasoning_content is an optional fixed native assistant prefill, not output.
    media_spans lists (start, hf_vision.MediaChunk) pairs for image branches:
    token_ids then holds each image's embeddings as request-unique negative
    placeholder IDs, so prefix comparison and usage accounting see expanded
    image tokens without inventing vocabulary IDs.

    render_only compilation leaves both ID lists empty and is not executable.
    frozen prevents attribute reassignment, not mutation of the contained lists.
    """

    branch_id: str
    token_ids: list[int]
    output_ids: list[int]
    messages: list[dict]
    answer_prefix: str
    reasoning_content: str | None = None
    media_spans: tuple = ()


@dataclass
class CompiledRequest:
    """Bind the semantic prompt plan to its HF-specific executable branches.

    Keep this pair together until response scoring: branch IDs and output order
    must be interpreted against the plan that created them. Branch order initially
    matches plan order; the backend may reorder execution for efficient padding.
    binary_noul_keys identifies temporary Choice questions that the HF response
    adapter must restore to Noul after common scores their no/yes logits.
    """

    plan: PromptPlan
    branches: list[Branch]
    binary_noul_keys: tuple[str, ...] = ()
    # hf_vision.RequestImages for image chat: owns the request's mtmd chunks
    # and encoded embeddings for as long as the compiled branches are alive.
    media: Any = None


def common_prefix(sequences):
    """Return the longest identical token prefix of a nonempty sequence group.

    The caller validates nonempty prompts. An empty shared prefix is valid.
    Comparing actual IDs is essential: matching rendered string fragments alone
    does not prove that the tokenizer produced reusable prefix tokens.
    """
    first = sequences[0]
    end = min(map(len, sequences))
    for other in sequences[1:]:
        for i in range(end):
            if first[i] != other[i]:
                end = i
                break
    return first[:end]


def render_chat(renderer, messages, **kwargs):
    """Preserve turns unless the native template requires strict alternation.

    Appending a classifier question to user-ending chat otherwise makes valid
    Gemma3 conversations unrenderable. For that native constraint only, coalesce
    adjacent user content in order, preserving every image and text block.
    """
    from jinja2.exceptions import TemplateError
    try:
        return renderer.apply_chat_template(messages, **kwargs)
    except TemplateError as exc:
        if 'alternat' not in str(exc).lower():
            raise ValueError('Model chat template rejected the conversation') from exc
        merged = []
        for message in messages:
            if merged and message['role'] == merged[-1]['role'] == 'user':
                left, right = merged[-1]['content'], message['content']
                if isinstance(left, str) and isinstance(right, str):
                    merged[-1]['content'] = left + '\n\n' + right
                else:
                    left = [{'type': 'text', 'text': left}] if isinstance(left, str) else left
                    right = [{'type': 'text', 'text': right}] if isinstance(right, str) else right
                    merged[-1]['content'] = left + [{'type': 'text', 'text': '\n\n'}] + copy.deepcopy(right)
            else:
                merged.append(copy.deepcopy(message))
        if len(merged) == len(messages):
            raise ValueError('Model chat template requires alternating conversation roles') from exc
        try:
            return renderer.apply_chat_template(merged, **kwargs)
        except TemplateError as retry:
            raise ValueError('Model chat template rejected the conversation') from retry


class PromptCompiler:
    """Render shared classifier plans with a model's native tokenizer template."""

    def __init__(self, tokenizer, max_tokens=16384, version=DEFAULT_TEMPLATE_VERSION, prompt_policy="baseline", max_choice_options=255, vision=None, max_image_width=None, max_image_height=None,
                 default_image_max_width=None, default_image_max_height=None):
        """Store the renderer, complete-prompt token limit, and shared version.

        Version validation is performed by common.prepare_prompt during compile.
        Named prompt policies are startup-selected HF formatting/scoring adaptations.
        vision is an optional hf_vision.MtmdVision (a loaded mmproj projector);
        without it, image chat is rejected. This class does not load a
        tokenizer or model on its own.
        """
        self.tokenizer = tokenizer
        from hf_vision import validate_image_resize_config
        validate_image_resize_config(max_image_width, max_image_height,
                                     default_image_max_width, default_image_max_height)
        self.vision = vision
        self.max_image_width = max_image_width
        self.max_image_height = max_image_height
        self.default_image_max_width = default_image_max_width
        self.default_image_max_height = default_image_max_height
        self.max_tokens = max_tokens
        self.version = version
        validate_policy(prompt_policy)
        self.prompt_policy = prompt_policy
        if type(max_choice_options) is not int or not 2 <= max_choice_options <= 255:
            raise ValueError("max_choice_options must be between 2 and 255")
        self.max_choice_options = max_choice_options
        self._extended_choice_labels = None

    async def compile_async(self, request):
        return await asyncio.to_thread(self.compile, request)

    def validate_choice_capacity(self):
        """Allocate fixed-width labels; every real prompt is checked again below."""
        if self.max_choice_options <= 50:
            return ()
        if self._extended_choice_labels is None:
            import itertools
            import string
            boundary = '{"answer": "'
            prefix = self.tokenizer.encode(boundary, add_special_tokens=False)
            labels, seen = [], set()
            special = set(getattr(self.tokenizer, 'all_special_ids', ()))
            for pair in itertools.product(string.ascii_uppercase, repeat=2):
                label = ''.join(pair)
                ids = self.tokenizer.encode(boundary + label, add_special_tokens=False)
                if (len(ids) == len(prefix) + 1 and ids[:-1] == prefix
                        and ids[-1] not in seen and ids[-1] not in special):
                    labels.append(label)
                    seen.add(ids[-1])
            if len(labels) < self.max_choice_options:
                raise ValueError(
                    f"Tokenizer supports only {len(labels)} distinct two-letter Choice labels; "
                    "reduce --max-choice-options (50 or fewer uses legacy labels)"
                )
            self._extended_choice_labels = tuple(labels)
        return self._extended_choice_labels

    def compile(self, request: ClassifierRequest, *, render_only=False):
        """Validate text/image input and compile one branch per shared-plan question.

        request may be a ClassifierRequest or an input dictionary. render_only
        returns roles/content and the shared plan for diagnostics/tests; it skips
        chat-template tokenization, token limits, and output-boundary checks.
        Such a result must not be sent to LlamaCppBackend.score.

        Unsupported media/tools, malformed token boundaries, and overlong prompts
        raise ValueError. Caller data is preserved when merging system messages.
        """
        if not isinstance(request, ClassifierRequest):
            request = ClassifierRequest.model_validate(request)
        # Reject unimplemented tools/options rather than silently dropping them.
        if request.tools or request.mm_processor_kwargs:
            raise ValueError(
                "This server does not support tools or native processor overrides"
            )
        from hf_vision import image_resize_bounds
        image_width, image_height = image_resize_bounds(
            request.media_io_kwargs, max_image_width=self.max_image_width,
            max_image_height=self.max_image_height,
            default_image_max_width=self.default_image_max_width,
            default_image_max_height=self.default_image_max_height,
        )
        if request.messages and any(
            m.model_extra or m.role in {"tool", "function"} or m.content is None
            for m in request.messages
        ):
            raise ValueError("This server does not support tool messages or null content")
        chat = [m.model_dump(exclude_none=True) for m in request.messages] if request.messages else None
        images, media = [], None
        if chat and any(not isinstance(m['content'], str) for m in chat):
            from hf_vision import image_messages
            chat, images = image_messages(chat, max_image_width=image_width,
                                          max_image_height=image_height)
            if images:
                if self.vision is None:
                    raise ValueError(
                        "Image chat requires a vision projector; start the server with --mmproj"
                    )
                if not render_only:
                    media = self.vision.prepare(images)
            # System policy assembly operates on text; image blocks are user-only.
            for message in chat:
                if isinstance(message['content'], list) and all(b['type'] == 'text' for b in message['content']):
                    message['content'] = ''.join(b['text'] for b in message['content'])
            if images:
                # llama-server convention: each image becomes a text block
                # holding the mtmd media marker, and mtmd supplies the model's
                # own image begin/end tokens. Image turns stay block lists, as
                # some templates (e.g. SmolVLM) render only list content.
                for message in chat:
                    if isinstance(message['content'], list):
                        message['content'] = [
                            {'type': 'text', 'text': self.vision.marker} if block['type'] == 'image' else block
                            for block in message['content']
                        ]

        largest_choice = max((len(q.criteria) for q in request.questions.values()
                              if q.type == 'choice'), default=0)
        if largest_choice > self.max_choice_options:
            raise ValueError(f"Choice has {largest_choice} options; maximum is {self.max_choice_options}")
        labels = self.validate_choice_capacity() if largest_choice > 50 else ()
        plan, binary_noul_keys = prepare_policy(
            request, self.version, self.prompt_policy, extended_choice_labels=labels
        )
        system = plan.system_prompt_prefix + plan.prefix_instruction
        # Some native templates (Gemma) put the thinking flag in the SYSTEM turn.
        # Shared policies therefore choose it once per request, never per branch.
        shared_thinking = any(q.type == 'choice' for q in request.questions.values())
        branches = []
        for question in plan.questions:
            content = plan.suffix_instruction + question.instruction
            # State is JSON-serialized, including quotes around string states.
            # For chat, preserve turn boundaries and merge only a leading system
            # turn; the final selected question is always a new user message.
            if request.messages is None:
                messages = [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": f"State:\n{canonical(request.state)}\n\n" + content,
                    },
                ]
            else:
                # Dump into new dictionaries so adding classifier instructions
                # never mutates the caller's existing conversation.
                messages = copy.deepcopy(chat)
                if messages[0]["role"] == "system":
                    messages[0]["content"] = system + "\n" + messages[0]["content"]
                else:
                    messages.insert(0, {"role": "system", "content": system})
                messages.append({"role": "user", "content": content})

            messages, reasoning_content = format_branch(
                messages, request, question.question_id, self.prompt_policy
            )
            ids, output_ids, spans = [], [], ()
            if not render_only:
                renderer = self.tokenizer
                # Render first, then append incomplete JSON to the open assistant
                # position. Do not create a completed assistant message or add
                # a closing brace/EOS before the next-token scoring position.
                if self.prompt_policy == "baseline":
                    text = (
                        render_chat(
                            renderer, messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        + question.answer_prefix
                    )
                else:
                    # Supply a fixed assistant prefill via the native template;
                    # this does not run a generation or reasoning stage.
                    assistant = {"role": "assistant", "content": question.answer_prefix}
                    if reasoning_content is not None:
                        assistant["reasoning_content"] = reasoning_content
                    # Match the native serving renderer's OpenAI text-block
                    # representation. Templates may distinguish a string from
                    # one text block (e.g. a system-turn boundary space). Do not
                    # hardcode model names, whitespace, or tokenizer IDs here.
                    native_messages = [
                        {**message, "content": ([{"type": "text", "text": message["content"]}]
                                                if isinstance(message['content'], str) else message['content'])}
                        for message in messages + [assistant]
                    ]
                    text = render_chat(
                        renderer, native_messages, tokenize=False,
                        add_generation_prompt=False, continue_final_message=True,
                        enable_thinking=(shared_thinking if self.prompt_policy.startswith('shared_')
                                         else reasoning_content is not None),
                    )
                    if reasoning_content is not None and reasoning_content not in text:
                        raise ValueError("Model chat template did not preserve fixed policy reasoning content")
                # The template already supplies special tokens. Adding another
                # BOS/EOS during encode would alter the intended model input.
                if media is not None:
                    ids, spans, tail = self.vision.tokenize(self.tokenizer, text, media)
                    tail_ids = self.tokenizer.encode(tail, add_special_tokens=False)
                else:
                    ids = self.tokenizer.encode(text, add_special_tokens=False)
                    tail, tail_ids = text, ids
                if not ids or len(ids) > self.max_tokens:
                    raise ValueError(
                        f"Branch for {question.question_id!r} must contain 1–{self.max_tokens} tokens"
                    )
                # Derive IDs at the actual rendered boundary, not from isolated
                # label encoding. Check every branch; templates/context can affect it.
                # Image branches check the text after their final image: mtmd
                # tokenizes each text span between images independently.
                for label in question.output_labels:
                    extended = self.tokenizer.encode(tail + label, add_special_tokens=False)
                    if len(extended) != len(tail_ids) + 1 or extended[:-1] != tail_ids:
                        raise ValueError(
                            f"Answer label {label!r} is not single-token stable"
                        )
                    output_ids.append(extended[-1])
                if len(set(output_ids)) != len(output_ids):
                    raise ValueError("Output labels must map to distinct token IDs")
            branches.append(
                Branch(
                    question.branch_id,
                    ids,
                    output_ids,
                    messages,
                    question.answer_prefix,
                    reasoning_content,
                    spans,
                )
            )
        return CompiledRequest(plan, branches, binary_noul_keys, media)


# Inference result


@dataclass
class BackendResult:
    """Results keyed by plan-local branch ID, independent of batch execution order.

    This server stores one label-to-float mapping per branch, keyed by the
    shared plan's exact output labels and ordered like the corresponding
    question's output_labels. The broad Any annotation also permits the same
    form from test backends, accepted by common's scorer.

    metrics contains internal timing/token/batch counters. branch_output_tokens
    is zero for logits-only inference. It does not supply input usage; the
    service computes the unique token-prefix union from the compiled prompts.
    """

    # Compact tensors in each plan question's output_labels order, keyed by ID.
    logits: dict[str, Any]
    metrics: dict


# Logical input-token accounting


def unique_prompt_tokens(sequences):
    """Return the number of distinct token-tree edges across all sequences.

    Example: [1, 2, 3] and [1, 2, 4] count as four tokens, not six. Duplicate
    paths add nothing; a path that ends inside another adds no new suffix. Empty
    input returns zero. Sorting creates a new list and leaves caller order intact.

    Lexicographic neighbors share the greatest already-counted prefix for each
    next sequence. Subtract that overlap from its length instead of building an
    explicit trie. No global cache or backend warmup information is consulted.
    """
    # Lexicographically adjacent sequences share the maximum previously seen
    # prefix. Count each token-tree edge once, including hierarchical prefixes.
    total = 0
    previous = []
    for sequence in sorted(sequences):
        shared = 0
        for left, right in zip(previous, sequence):
            if left != right:
                break
            shared += 1
        total += len(sequence) - shared
        previous = sequence
    return total


# Shared-prefix model execution


class LlamaCppBackend:
    """Own one llama.cpp context and serialize all decodes through a thread lock.

    The context is a unified KV pool of n_ctx cells shared by sequence ids:
    sequence 0 holds the request's common prefix; each suffix-batch row runs on
    its own sequence id 1..N after a llama_memory_seq_cp copies the prefix cells.
    Rows are packed into one llama_decode call per batch (no padding tokens),
    with logits requested only at each row's final suffix position.
    """

    def __init__(
        self, model, context, *, max_batch_size=32, max_batch_tokens=32768,
        share_prefix=True, vision=None,
    ):
        """Bind one loaded model/context and bounds on packed suffix batches.

        max_batch_size caps rows per decode and sequence ids (1 + rows must fit
        the context's n_seq_max). max_batch_tokens caps the packed suffix token
        count per decode; it does not bound the prefix prefill. The compiler
        enforces the full model-context limit separately.
        """
        if max_batch_size < 1 or max_batch_tokens < 1:
            raise ValueError("Batch limits must be positive")
        self.model = model
        self.context = context
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        # False makes every branch prefill its own full prompt instead of
        # copying shared prefix cells, the only strategy that is valid when
        # llama.cpp cannot duplicate an architecture's rolling state.
        self.share_prefix = share_prefix
        # Optional hf_vision.MtmdVision bound to this model; required only to
        # decode image spans, whose embeddings it encodes once per request.
        self.vision = vision
        # Protect the context even if cancellation returns before its thread exits.
        self._lock = threading.Lock()

    async def score(self, compiled):
        """Run blocking engine work in a worker thread with cooperative cancellation.

        Cancelling an asyncio task cannot interrupt an in-flight GPU kernel.
        Set a stop flag for the worker to check before its next decode, then let
        cancellation propagate. The context lock remains held until the worker exits.
        """
        stop = threading.Event()
        try:
            return await asyncio.to_thread(self._score, compiled, stop)
        except asyncio.CancelledError:
            stop.set()
            raise

    def _decode_rows(self, llama_cpp, rows):
        """Pack rows into one llama_batch and decode; return flagged positions.

        rows are (seq_id, pos_start, token_ids, want_last_logits) tuples. Every
        field of every used slot is explicitly initialized: llama_batch memory
        is recycled by the allocator and may otherwise contain garbage. Batch
        arrays are written through one cached handle each to avoid re-fetching
        the ctypes pointers for every element.
        """
        n_tokens = sum(len(tokens) for _, _, tokens, _ in rows)
        batch = llama_cpp.llama_batch_init(n_tokens, 0, 1)
        try:
            batch.n_tokens = n_tokens
            token, pos = batch.token, batch.pos
            seq_id, n_seq_id, logits = batch.seq_id, batch.n_seq_id, batch.logits
            index = 0
            flagged = []
            for seq, start, tokens, want in rows:
                last = len(tokens) - 1
                for offset, token_id in enumerate(tokens):
                    token[index] = int(token_id)
                    pos[index] = start + offset
                    seq_id[index][0] = int(seq)
                    n_seq_id[index] = 1
                    logits[index] = bool(want and offset == last)
                    index += 1
                if want:
                    flagged.append(index - 1)
            if llama_cpp.llama_decode(self.context.ctx, batch) != 0:
                raise RuntimeError("llama_decode failed for the scoring batch")
            return flagged
        finally:
            llama_cpp.llama_batch_free(batch)

    def _decode_range(self, llama_cpp, compiled, branch, begin, end, seq):
        """Decode token_ids[begin:end] of one branch into seq, without logits.

        Text runs go through one packed decode each; image spans decode their
        (once-per-request encoded) projector embeddings through mtmd. begin
        and end never cut an image span. Returns the number of decodes issued.
        """
        ids, spans = branch.token_ids, branch.media_spans
        pos = position_at(spans, begin)
        index, decodes = begin, 0
        for start, chunk in spans:
            if start < begin or start >= end:
                continue
            if start > index:
                self._decode_rows(llama_cpp, [(seq, pos, ids[index:start], False)])
                pos += start - index
                decodes += 1
            pos = self.vision.decode(
                self.context.ctx, compiled.media, chunk, pos, seq,
                llama_cpp.llama_n_batch(self.context.ctx),
            )
            index = start + chunk.n_tokens
            decodes += 1
        if end > index:
            self._decode_rows(llama_cpp, [(seq, pos, ids[index:end], False)])
            decodes += 1
        return decodes

    def _score(self, compiled, stop):
        """Execute one request; all model/KV work stays under the same lock."""
        import llama_cpp

        with self._lock:
            if stop.is_set():
                raise asyncio.CancelledError()
            start = time.perf_counter()
            branches = compiled.branches
            sequences = [b.token_ids for b in branches]
            if not sequences or any(not ids for ids in sequences):
                raise ValueError("Expected nonempty scoring prompts")
            if compiled.plan is None:
                raise ValueError("Backend scoring requires the compiled plan")
            multimodal = any(b.media_spans for b in branches)
            if multimodal and (self.vision is None or compiled.media is None):
                raise ValueError("Image branches require a loaded vision projector")
            labels = {
                question.branch_id: question.output_labels
                for question in compiled.plan.questions
            }
            # Leave at least one suffix token, including for identical prompts.
            prefix_len = (
                len(common_prefix(sequences)[: min(map(len, sequences)) - 1])
                if self.share_prefix
                else 0
            )
            # Image placeholders are request-unique, so an equal token prefix
            # also means equal images. Never split one image's embeddings
            # between the shared prefix and a row: cut back to its start.
            for branch in branches:
                for span_start, chunk in branch.media_spans:
                    if span_start < prefix_len < span_start + chunk.n_tokens:
                        prefix_len = span_start
            prefix_pos = position_at(branches[0].media_spans, prefix_len)
            memory = self.context.memory
            # Start every request from an empty KV pool: prefix work never
            # persists across requests, so results cannot leak between callers.
            llama_cpp.llama_memory_clear(memory, True)
            decodes = 0
            if prefix_len:
                # One unchunked prefill decode on sequence 0 (plus one per image
                # span). No logits are needed; scoring positions are in suffixes.
                decodes += self._decode_range(llama_cpp, compiled, branches[0], 0, prefix_len, 0)
            lengths = [len(ids) - prefix_len for ids in sequences]
            if max(lengths) > self.max_batch_tokens:
                raise ValueError("A question suffix exceeds max_batch_tokens")
            # Each row's suffix splits at the end of its last image: anything
            # before that is decoded per row, the text after it is packed.
            splits = [
                max(
                    [s + c.n_tokens for s, c in b.media_spans if s >= prefix_len],
                    default=prefix_len,
                )
                for b in branches
            ]
            # Longest first groups equal lengths together. Results carry branch
            # IDs, so execution order need not equal the original question order.
            pending = sorted(
                range(len(sequences)), key=lambda i: lengths[i], reverse=True
            )
            results = {}
            sizes = []
            packed_tokens = 0
            n_ctx = self.context.n_ctx()
            while pending:
                if stop.is_set():
                    raise asyncio.CancelledError()
                # Width-based limits mirror the previous engine's padded-batch
                # accounting: rows times width bounded by max_batch_tokens, and
                # every live KV cell (prefix plus suffixes) bounded by n_ctx.
                width = lengths[pending[0]]
                limit = min(
                    self.max_batch_size,
                    self.max_batch_tokens // width if width else 1,
                    max(1, (n_ctx - prefix_len) // width) if width else 1,
                )
                batch, pending = pending[:limit], pending[limit:]
                # Each row runs on its own sequence id sharing the prefix cells:
                # the unified cache marks copied cells for both sequences, so
                # attention sees prefix+suffix without recomputing the prefill.
                for row, i in enumerate(batch):
                    if prefix_len:
                        llama_cpp.llama_memory_seq_cp(memory, 0, row + 1, 0, prefix_pos)
                    if splits[i] > prefix_len:
                        if stop.is_set():
                            raise asyncio.CancelledError()
                        decodes += self._decode_range(
                            llama_cpp, compiled, branches[i], prefix_len, splits[i], row + 1
                        )
                rows = [
                    (
                        row + 1,
                        position_at(branches[i].media_spans, splits[i]),
                        sequences[i][splits[i]:],
                        True,
                    )
                    for row, i in enumerate(batch)
                ]
                flagged = self._decode_rows(llama_cpp, rows)
                # One flagged output per row, in batch order: read exactly the
                # permitted label logits and copy them to plain Python floats.
                for row, i in enumerate(batch):
                    branch = branches[i]
                    logits = llama_cpp.llama_get_logits_ith(
                        self.context.ctx, flagged[row]
                    )
                    results[branch.branch_id] = {
                        label: float(logits[token_id])
                        for label, token_id in zip(
                            labels[branch.branch_id], branch.output_ids
                        )
                    }
                # Row sequences are discarded after each batch: seq_rm drops
                # their claim on the shared prefix cells and frees their suffix
                # cells, making both the cells and the sequence ids reusable.
                for row in range(len(batch)):
                    llama_cpp.llama_memory_seq_rm(memory, row + 1, 0, -1)
                decodes += 1
                sizes.append(len(batch))
                packed_tokens += sum(lengths[i] for i in batch)
            metrics = {
                "backend": "llama-cpp",
                "prefill_strategy": (
                    "shared_prefix" if self.share_prefix else "per_branch"
                ),
                "prefix_tokens": prefix_len,
                "suffix_batch_sizes": sizes,
                "engine_forwards": decodes,
                # These are distinct accounting views, not interchangeable:
                # branch_prompt_tokens repeats shared context per branch;
                # computed_prompt_tokens counts cells the engine filled;
                # logical_prefill_tokens counts each branch's suffix once
                # beyond the single shared prefix. Rows are packed without
                # padding tokens, so the latter two views agree here.
                "branch_prompt_tokens": sum(map(len, sequences)),
                "computed_prompt_tokens": prefix_len + packed_tokens,
                "logical_prefill_tokens": prefix_len + sum(lengths),
                "padded_suffix_tokens": packed_tokens,
                "branch_output_tokens": 0,
                "scored_positions": len(sequences),
                "backend_seconds": time.perf_counter() - start,
            }
            if multimodal:
                shared = all(
                    s + c.n_tokens <= prefix_len
                    for b in branches for s, c in b.media_spans
                )
                metrics.update({
                    # Every image is encoded once per request; "independent"
                    # decodes those embeddings into each branch separately.
                    "prefill_strategy": (
                        "multimodal_shared_prefix" if shared else "multimodal_independent"
                    ),
                    "vision_forwards": len(compiled.media.embeddings),
                    "image_decodes": (
                        len(branches[0].media_spans) if shared
                        else sum(len(b.media_spans) for b in branches)
                    ),
                })
            return BackendResult(results, metrics)


def position_at(spans, index):
    """Return the decoder position of token_ids[index] for a branch's image spans.

    Text and non-M-RoPE images take one position per token; an M-RoPE image
    (Qwen-VL family) takes chunk.n_pos positions for its n_tokens embeddings.
    index must not fall inside an image span.
    """
    offset = 0
    for start, chunk in spans:
        if start + chunk.n_tokens <= index:
            offset += chunk.n_tokens - chunk.n_pos
    return index - offset


# Request admission and response assembly


def validate_rope_factor(factor):
    if not math.isfinite(factor) or factor < 1:
        raise ValueError("RoPE factor must be finite and at least 1")


def configure_rope(context_params, factor):
    """Configure llama.cpp linear position interpolation on the context.

    llama.cpp applies rope_freq_scale = 1/factor as its linear rope scaling:
    position p then uses the original rotary angle at p/factor, matching the
    engine's --rope-scale behavior. The admission limit stays governed by
    --max-model-len; extending positional capacity does not establish accuracy,
    and interpolation also changes short-input behavior.
    """
    validate_rope_factor(factor)
    if factor == 1:
        return
    context_params.rope_scaling_type = 1  # LLAMA_ROPE_SCALING_TYPE_LINEAR
    context_params.rope_freq_scale = 1.0 / float(factor)


def extend_laya_rope(agent, factor):
    """Experimental linear position interpolation for Laya's ModernBERT encoder.

    Dividing both full/sliding inverse frequencies by two maps position p to
    its original rotary angle at p/2. The local attention window is unchanged.
    This changes short-input behavior too; it does not establish longer-context
    accuracy. Refuse other architectures or already-scaled RoPE rather than
    silently composing incompatible scaling rules.
    """
    if factor == 1:
        return
    validate_rope_factor(factor)
    encoder = agent.model.encoder
    rotary = getattr(encoder, "rotary_emb", None)
    if encoder.config.model_type != "modernbert" or rotary is None:
        raise ValueError("Laya RoPE extension currently requires ModernBERT")
    if getattr(agent, "_simple_jev_rope_extended", False):
        raise ValueError("Laya RoPE has already been extended")
    kinds = ("full_attention", "sliding_attention")
    for kind in kinds:
        if rotary.rope_type.get(kind) != "default" or not hasattr(
            rotary, f"{kind}_inv_freq"
        ):
            raise ValueError("Laya RoPE extension requires unscaled full/sliding RoPE")
    target = int(int(agent.cfg["max_len"]) * factor)
    if target > encoder.config.max_position_embeddings:
        raise ValueError("Extended Laya sequence exceeds encoder position capacity")
    import torch

    with torch.no_grad():
        for kind in kinds:
            getattr(rotary, f"{kind}_inv_freq").div_(factor)
            getattr(rotary, f"{kind}_original_inv_freq").div_(factor)
    agent.cfg["max_len"] = target
    agent._simple_jev_rope_extended = True


class LayaBackend:
    """Native encoder adapter; no chat prefill, vocabulary labels, or KV cache.

    The SDK owns option-marker formatting and temperature scaling. We retain its
    binary Noul probability, normalize confidence to Simple Jev's max probability,
    and omit SDK-only action fields. A lock protects the model even if the HTTP
    coroutine is cancelled while its worker is finishing.
    """

    def __init__(self, agent, max_tokens):
        self.agent = agent
        self.max_tokens = min(max_tokens, int(agent.cfg["max_len"]))
        self._lock = threading.Lock()

    async def classify_native(self, request):
        stop = threading.Event()
        try:
            return await asyncio.to_thread(self._classify, request, stop)
        except asyncio.CancelledError:
            stop.set()
            raise

    def _classify(self, request, stop):
        import math
        from laya.common import build_sequence, serialize_state

        with self._lock:
            if stop.is_set():
                raise asyncio.CancelledError()
            if (
                request.tools
                or request.mm_processor_kwargs
                or request.media_io_kwargs
            ):
                raise ValueError("Laya supports text state and text chat only")
            if request.options.raw_logits:
                raise ValueError("Laya raw_logits diagnostics are not supported")
            state = request.state
            if request.messages is not None:
                # Chat is serialized as role/content data, not a causal chat template.
                turns = []
                for message in request.messages:
                    if (
                        not isinstance(message.content, str)
                        or message.role not in {"system", "user", "assistant"}
                        or message.model_extra
                    ):
                        raise ValueError(
                            "Laya supports plain system/user/assistant text messages only"
                        )
                    turns.append({"role": message.role, "content": message.content})
                state = turns
            questions = {key: q.model_dump() for key, q in request.questions.items()}
            # SDK truncates state by default. Probe with a generous token budget
            # and reject overflow so important context cannot disappear silently.
            for key, definition in questions.items():
                q = self.agent._to_internal(definition)
                state_size = len(
                    self.agent.tok(
                        serialize_state(state).replace(self.agent.tok.mask_token, " "),
                        add_special_tokens=False,
                    )["input_ids"]
                )
                ids, _ = build_sequence(
                    self.agent.tok,
                    state,
                    q,
                    state_size + self.agent.cfg.get("head_max_len", 192) + 65536,
                    self.agent.cfg.get("head_max_len", 192),
                )
                if len(ids) > self.max_tokens:
                    raise ValueError(
                        f"Laya question {key!r} exceeds {self.max_tokens} input tokens"
                    )
            if stop.is_set():
                raise asyncio.CancelledError()
            result = self.agent.predict(state, questions)
            if set(result["answers"]) != set(questions):
                raise ValueError("Laya returned incomplete answers")
            answers = {}
            for key, q in questions.items():
                source = result["answers"][key]
                if q["type"] == "noul":
                    value = float(source["noul"])
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("Laya returned an invalid probability")
                    answers[key] = {"type": "noul", "noul": value}
                    continue
                labels = (
                    list(q["criteria"])
                    if q["type"] == "choice"
                    else [str(i) for i in range(len(q["criteria"]))]
                )
                probs = source["probabilities"]
                if (
                    set(probs) != set(labels)
                    or any(
                        not math.isfinite(float(v)) or not 0 <= float(v) <= 1
                        for v in probs.values()
                    )
                    or abs(sum(probs.values()) - 1) > 0.01
                ):
                    raise ValueError("Laya returned invalid option probabilities")
                answer = {
                    "type": q["type"],
                    "probabilities": probs,
                    "confidence": max(probs.values()),
                }
                if q["type"] == "choice":
                    if source["choice"] not in labels:
                        raise ValueError("Laya returned an invalid choice")
                    answer["choice"] = source["choice"]
                else:
                    value = float(source["score"])
                    if not math.isfinite(value) or not 0 <= value <= len(labels) - 1:
                        raise ValueError("Laya returned an invalid score")
                    answer.update(
                        score=value,
                        legend={str(i): c for i, c in enumerate(q["criteria"])},
                    )
                answers[key] = answer
            return {
                "model": request.model,
                "answers": answers,
                "usage": result["usage"],
            }


class OverloadedError(Exception):
    """Admission capacity is exhausted; the HTTP layer translates this to 429."""


class DecisionService:
    """Manage one configured model, its compiler/backend, and request lifecycle.

    Use on one asyncio event loop: the admission counter is intentionally updated
    without awaiting between its capacity check and increment. Compiler/backend
    dependencies allow service tests to run without model weights.
    """

    def __init__(
        self,
        model,
        compiler,
        backend,
        *,
        metadata=None,
        concurrency=4,
        queue_size=16,
        max_request_branches=100,
        model_aliases=(),
        enforce_model_id=False,
        max_choice_options=255,
        advanced_metrics=None,
    ):
        """Configure admission and diagnostic output for a loaded model.

        concurrency counts active coroutine slots; queue_size adds waiting slots.
        Supply a positive concurrency and nonnegative queue size. model_aliases
        permits additional names when enforce_model_id is enabled, not dynamic
        loading. By default any request ID is accepted; responses identify model.
        max_choice_options limits Choice cardinality independently of branch count.
        advanced_metrics explicitly overrides the environment flag when provided;
        otherwise 1/true/yes/on enable ENABLE_OPEN_JEV_ADVANCED_METRICS.
        """
        if max_request_branches < 1:
            raise ValueError("max_request_branches must be positive")
        self.max_request_branches = max_request_branches
        if type(max_choice_options) is not int or not 2 <= max_choice_options <= 255:
            raise ValueError("max_choice_options must be between 2 and 255")
        self.max_choice_options = max_choice_options
        self.enforce_model_id = enforce_model_id
        self.model_aliases = {model, *model_aliases}
        self.model = model
        self.compiler = compiler
        self.backend = backend
        self.metadata = metadata or {}
        self.advanced_metrics = (
            os.environ.get("ENABLE_OPEN_JEV_ADVANCED_METRICS", "").strip().lower()
            in {"1", "true", "yes", "on"}
            if advanced_metrics is None
            else advanced_metrics
        )
        self._semaphore = asyncio.Semaphore(concurrency)
        self._capacity = concurrency + queue_size
        self._inflight = 0

    async def classify(self, request):
        """Return a complete response or raise validation/overload/cancellation.

        Validate and admit before expensive work. Admission is released in finally
        on success, failure, or cancellation. The queue timer ends when the active
        slot is acquired; total_seconds spans admitted work through response build.
        These timings use a monotonic clock and are not distributed trace spans.
        """
        if not isinstance(request, ClassifierRequest):
            request = ClassifierRequest.model_validate(request)
        if self.enforce_model_id and request.model not in self.model_aliases:
            raise ValueError(f"Served model is {self.model!r}")
        for question in request.questions.values():
            if question.type == 'choice' and len(question.criteria) > self.max_choice_options:
                raise ValueError(f"Choice options exceed configured maximum of {self.max_choice_options}")
        request = request.model_copy(update={'model': self.model})
        # v1 has exactly one inference branch per question; candidate count no
        # longer expands requests. The shared schema separately caps 256 questions.
        branches = len(request.questions)
        if branches > self.max_request_branches:
            raise ValueError(
                f"Request has {branches} scoring branches; maximum is {self.max_request_branches}"
            )
        if self._inflight >= self._capacity:
            raise OverloadedError("Scoring queue is full")
        # No await between this check/increment pair: other tasks on the same
        # loop cannot interleave admission and oversubscribe the capacity.
        self._inflight += 1
        start = time.perf_counter()
        try:
            async with self._semaphore:
                queued = time.perf_counter() - start
                if hasattr(self.backend, "classify_native"):
                    response = await self.backend.classify_native(request)
                    response['model'] = self.model
                    if self.advanced_metrics:
                        response["metadata"] = {
                            **self.metadata,
                            "format": "laya-native",
                            "usage_accounting": "sum_of_question_sequence_tokens",
                        }
                        response["metrics"] = {
                            "queue_seconds": queued,
                            "total_seconds": time.perf_counter() - start,
                        }
                    return response
                # Tokenization can be expensive and must not block cancellation/HTTP.
                if hasattr(self.compiler, "compile_async"):
                    compiled = await self.compiler.compile_async(request)
                else:
                    if request.messages and any(
                        not isinstance(m.content, str) or m.role in {"tool", "function"}
                        for m in request.messages
                    ):
                        raise ValueError(
                            "Multimodal/tool chat requires a native renderer; this backend accepts text chat only"
                        )
                    compiled = await asyncio.to_thread(self.compiler.compile, request)
                # The compiler retains the shared plan alongside executable IDs.
                # Never reconstruct label meaning from model output or batch order.
                result = await self.backend.score(compiled)
                response = build_response(
                    compiled.plan,
                    result.logits,
                    # Logical prefix-union accounting excludes batch padding and
                    # repeated context, not a sum of the backend's forward sizes.
                    input_tokens=unique_prompt_tokens(
                        [b.token_ids for b in compiled.branches]
                    ),
                    output_tokens=result.metrics.get("branch_output_tokens", 0),
                    advanced=self.advanced_metrics,
                )
                restore_binary_noul(response, compiled.binary_noul_keys)
                # Common handles answer filtering and authoritative version data;
                # the server only adds backend-specific metadata and timings.
                if self.advanced_metrics:
                    response["metadata"] = {
                        **self.metadata,
                        **response["metadata"],
                        "usage_accounting": "unique_token_prefixes_and_engine_leaf_outputs",
                    }
                    response["metrics"] = {
                        **result.metrics,
                        "queue_seconds": queued,
                        "total_seconds": time.perf_counter() - start,
                    }
                return response
        finally:
            # Release service admission even if a cancelled worker is finishing.
            # The backend's lock still prevents overlapping access to the model.
            self._inflight -= 1


# HTTP routes and errors


def validation_response(message=None, errors=()):
    """Format explicit text or Pydantic errors without returning input payloads.

    Keep at most ten structured details and report the count of additional
    errors. Convert location tuples into readable dotted paths with array indices,
    omitting the leading transport-specific 'body' component. The first location
    becomes the top-level param; errors without a location use null.
    """
    details = []
    for error in errors[:10]:
        path = ""
        for part in error.get("loc", ()):
            if part == "body" and not path:
                continue
            if isinstance(part, int):
                path += f"[{part}]"
            else:
                path += ("." if path else "") + str(part)
        details.append(
            {"param": path or None, "message": error["msg"], "type": error["type"]}
        )
    if message is None:
        message = (
            "; ".join(
                f"{e['param']}: {e['message']}" if e["param"] else e["message"]
                for e in details
            )
            or "Invalid classifier request"
        )
        if len(errors) > len(details):
            message += f"; {len(errors) - len(details)} additional validation errors"
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": 422,
                "param": details[0]["param"] if details else None,
                "details": details,
            }
        },
    )


class ClassifierRoute(APIRoute):
    """Normalize errors only for classifier routes, leaving host handlers alone.

    Request parsing can fail before the endpoint function runs, so normalization
    belongs around FastAPI's generated route handler as well as in the endpoint.
    """

    def get_route_handler(self):
        """Wrap FastAPI parsing/execution while preserving non-422 exceptions."""
        handler = super().get_route_handler()

        async def validated(request):
            """Intercept classifier validation errors before they leave this route."""
            try:
                return await handler(request)
            except RequestValidationError as exc:
                # Exclude input values and exception contexts from public errors.
                return validation_response(errors=exc.errors())
            except HTTPException as exc:
                if exc.status_code == 422:
                    return validation_response(message=str(exc.detail))
                raise

        return validated


def create_app(service):
    """Build a standalone app around an already-created service.

    The CLI loads the model before calling this function. Readiness therefore
    reports that configured service, not a new inference probe on every request.
    """
    app = FastAPI(title="Simple-JEV", version="0.1.0")

    @app.get("/health")
    async def health():
        """Return the configured model identifier without invoking inference."""
        return {"status": "ready", "model": service.model}

    @app.get("/v1/models")
    async def models():
        """Discovery is metadata-only and never occupies an inference slot."""
        return {"object": "list", "data": [{
            "id": service.model, "object": "model", "created": 0,
            "owned_by": "simple-jev",
            "x_max_choice_options": service.max_choice_options,
        }]}

    attach_routes(app, lambda request: service)
    return app


def attach_routes(app, get_service):
    """Attach classifier endpoints; the alias stays out of generated OpenAPI.

    get_service is synchronous and request-scoped, allowing an embedding app to
    select its service without changing the classifier handler's implementation.
    """
    router = APIRouter(route_class=ClassifierRoute)

    @router.post("/v1/classifier")
    @router.post("/v1/systemone", include_in_schema=False)
    async def classify(body: ClassifierRequest, request: Request):
        """Run one service task and cancel it if the HTTP client disconnects."""
        task = asyncio.create_task(get_service(request).classify(body))
        try:
            # Wait with a timeout rather than awaiting task directly, so a long
            # tokenization/model pass does not prevent disconnect checks.
            while not task.done():
                await asyncio.wait({task}, timeout=0.1)
                if await request.is_disconnected():
                    task.cancel()
                    raise HTTPException(499, "Client disconnected")
            return await task
        except OverloadedError as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "1"}) from exc
        except ValidationError as exc:
            return validation_response(errors=exc.errors())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            # Always observe the child task's completion/exception. Cancelling
            # its coroutine does not forcibly stop an active model worker thread;
            # the backend implements cooperative stopping and a model lock.
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    app.include_router(router)


# Model loading and command-line entry point


def resolve_gguf_path(model_name, revision, gguf_file=None):
    """Return one local GGUF file path for a path, directory, or HF repository.

    A directory must contain exactly one .gguf file (sharded files following
    llama.cpp's <name>-00001-of-000NN.gguf scheme load together and count as
    one). gguf_file selects one file from a directory or Hugging Face repository.
    A Hugging Face repository is downloaded with huggingface_hub, keeping the
    repository identifier as the request-visible model name.
    """
    path = Path(model_name)
    if path.is_file():
        if gguf_file:
            raise ValueError("--gguf-file cannot be used with a GGUF file path")
        return str(path)
    if path.is_dir():
        matches = sorted(path.glob(gguf_file or "*.gguf"))
        if not gguf_file:
            matches = [m for m in matches if not is_mmproj(m.name)]
    else:
        if gguf_file:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ImportError(
                    "Downloading GGUF weights requires huggingface-hub; install it "
                    "or point --model at a local .gguf file"
                ) from exc
            if Path(gguf_file).name != gguf_file or not gguf_file.endswith(".gguf"):
                raise ValueError("--gguf-file must name one .gguf file")
            return hf_hub_download(
                repo_id=model_name, filename=gguf_file, revision=revision
            )
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "Downloading GGUF weights requires huggingface-hub; install it "
                "or point --model at a local .gguf file"
            ) from exc
        root = Path(snapshot_download(model_name, revision=revision, allow_patterns=["*.gguf"]))
        matches = [m for m in sorted(root.glob("*.gguf")) if not is_mmproj(m.name)]
        # Shards named <name>-00001-of-000NN.gguf are loaded as one model from
        # their first file; report the shard set, not each file, as the choice.
        shards = [m for m in matches if "-of-" in m.name]
        if shards:
            first = shards[0]
            matches = [first]
    if not matches:
        raise ValueError(f"No .gguf file found for model {model_name!r}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple .gguf files found for {model_name!r}; pass one file path"
        )
    return str(matches[0])


def is_mmproj(filename):
    """Vision projectors ship beside the text GGUF (e.g. mmproj-F16.gguf)."""
    return Path(filename).name.lower().startswith("mmproj")


def resolve_mmproj_path(mmproj, model_name, gguf_path, revision):
    """Return a local vision-projector GGUF for --mmproj.

    Accepts a local file path, a file name next to the resolved text GGUF (or
    inside a local --model directory), or a file name in the --model Hugging
    Face repository, which is then downloaded on its own.
    """
    candidate = Path(mmproj)
    if candidate.is_file():
        return str(candidate)
    if Path(mmproj).name != mmproj or not mmproj.endswith(".gguf"):
        raise ValueError("--mmproj must be a .gguf file path or a file name in the model repository")
    for folder in (Path(gguf_path).parent, Path(model_name)):
        if (folder / mmproj).is_file():
            return str(folder / mmproj)
    if Path(model_name).exists():
        raise ValueError(f"--mmproj file {mmproj!r} was not found next to the model")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=model_name, filename=mmproj, revision=revision)


def resolve_n_gpu_layers(device, n_gpu_layers):
    """Map the CLI device selection to llama.cpp layer offloading.

    None means "derive from device": cpu disables offload entirely, while the
    accelerator spellings (auto/gpu/vulkan/cuda/rocm/metal) offload every layer
    so llama.cpp routes through any compiled backend, including Vulkan. An
    explicit n_gpu_layers value always wins over --device.
    """
    if n_gpu_layers is not None:
        if n_gpu_layers < -1:
            raise ValueError("n_gpu_layers must be -1 or nonnegative")
        return n_gpu_layers
    normalized = device.strip().lower()
    if normalized == "cpu":
        return 0
    if normalized in {"auto", "gpu", "vulkan", "cuda", "rocm", "metal"}:
        return -1
    raise ValueError(
        f"Unknown device {device!r}; use cpu, auto, gpu, or an explicit --n-gpu-layers"
    )


KV_CACHE_TYPES = {"float32": 0, "float16": 1, "bfloat16": 30}
# ggml_type enum: F32=0, F16=1, BF16=30 in this llama.cpp release. GGUF weights
# keep their own quantization; --dtype selects only the KV cache precision.


class LlamaCppTokenizer:
    """Adapter from the HF-style tokenizer surface to one llama.cpp GGUF vocab.

    PromptCompiler needs exactly two calls: apply_chat_template(tokenize=False)
    and encode(text, add_special_tokens=False). Both are served natively by
    llama.cpp — its tokenizer reads the GGUF's embedded vocabulary, and the
    chat template is rendered from the GGUF's tokenizer.chat_template metadata
    with the same Jinja environment settings llama-cpp-python uses (matching
    Transformers' trim_blocks/lstrip_blocks conventions). Named prompt policies
    additionally render with continue_final_message, reproduced here with
    Transformers' own tag-and-truncate rule, and extended Choice labels
    consult all_special_ids.
    """

    def __init__(self, model, metadata=None, chat_template=None):
        """Bind a loaded GGUF vocabulary; chat_template overrides the embedded one."""
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        self._model = model
        metadata = metadata if metadata is not None else model.metadata()
        template = chat_template or metadata.get("tokenizer.chat_template")
        if not template:
            raise ValueError(
                "GGUF file has no tokenizer.chat_template metadata; the shared "
                "v1 template requires a chat-formatted model (or --chat-template-file)"
            )
        bos_text, eos_text = "", ""
        try:
            bos_id = int(metadata.get("tokenizer.ggml.bos_token_id", ""))
            eos_id = int(metadata.get("tokenizer.ggml.eos_token_id", ""))
            bos_text = model.token_get_text(bos_id) if 0 <= bos_id else ""
            eos_text = model.token_get_text(eos_id) if 0 <= eos_id else ""
        except ValueError:
            pass
        self._template = template
        self._bos_text = bos_text
        self._eos_text = eos_text
        self._formatters = {}
        self._special_ids = None

    def _formatter(self, add_generation_prompt):
        """Build (and cache) a Jinja formatter per generation-prompt setting.

        The formatter renders with its own constructor flag, so one cached
        instance exists for each add_generation_prompt value the compiler uses.
        """
        if add_generation_prompt not in self._formatters:
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter

            self._formatters[add_generation_prompt] = Jinja2ChatFormatter(
                template=self._template,
                eos_token=self._eos_text,
                bos_token=self._bos_text,
                add_generation_prompt=add_generation_prompt,
            )
        return self._formatters[add_generation_prompt]

    @property
    def all_special_ids(self):
        """Control-token IDs, the GGUF counterpart of HF's special tokens.

        Extended Choice labels must never map to one of these. Scanned once on
        first use; ordinary requests with 50 or fewer options never need it.
        """
        if self._special_ids is None:
            import llama_cpp

            control = llama_cpp.LLAMA_TOKEN_ATTR_CONTROL
            vocab = llama_cpp.llama_model_get_vocab(self._model.model)
            self._special_ids = frozenset(
                token
                for token in range(llama_cpp.llama_vocab_n_tokens(vocab))
                if llama_cpp.llama_vocab_get_attr(vocab, token) & control
            )
        return self._special_ids

    def encode(self, text, add_special_tokens=False):
        """Tokenize text with llama.cpp's native GGUF tokenizer.

        add_special_tokens=False keeps the template responsible for BOS/EOS:
        the rendered chat text already carries its special tokens, and adding
        another BOS would alter the intended model input.
        """
        return self._model.tokenize(
            text.encode("utf-8"), add_bos=bool(add_special_tokens), special=True
        )

    def apply_chat_template(self, messages, **kwargs):
        """Render the GGUF chat template exactly as the compiler requests it.

        tokenize must stay False: the compiler tokenizes the rendered text.
        add_generation_prompt opens the assistant turn; enable_thinking and
        any other Transformers-style flags are forwarded into the Jinja render,
        the same forwarding Transformers performs for its thinking-capable
        models.
        """
        if kwargs.get("tokenize"):
            raise ValueError("This adapter renders chat templates as text only")
        add_generation_prompt = bool(kwargs.get("add_generation_prompt"))
        continue_final = bool(kwargs.get("continue_final_message"))
        if continue_final and add_generation_prompt:
            raise ValueError(
                "continue_final_message and add_generation_prompt are not compatible"
            )
        # continue_final_message is a renderer option, not a template variable.
        forward = {
            key: value
            for key, value in kwargs.items()
            if key not in {"tokenize", "add_generation_prompt", "continue_final_message"}
        }
        formatter = self._formatter(add_generation_prompt)
        if continue_final:
            messages, final = mark_final_message(messages, self._template)
        try:
            rendered = formatter(messages=messages, **forward).prompt
        except ValueError as exc:
            # llama-cpp-python's raise_exception() raises ValueError; surface it
            # as Transformers does, a TemplateError, so render_chat can apply
            # its strict-alternation fallback (e.g. Gemma 3) unchanged.
            from jinja2.exceptions import TemplateError

            raise TemplateError(str(exc)) from exc
        if continue_final:
            rendered = cut_at_final_message(rendered, final)
        return rendered


# Transformers' sentinel for continue_final_message, reproduced verbatim.
CONTINUE_FINAL_MESSAGE_TAG = "CONTINUE_FINAL_MESSAGE_TAG "


def mark_final_message(messages, template):
    """Copy messages, appending the continuation tag to the final text.

    Mirrors Transformers 5.x render_jinja_template: the tag goes after the last
    text block of the final message (or after string content) so the rendered
    chat can be cut exactly where that message ends, before any end-of-turn
    tokens the template would emit.
    """
    import copy

    messages = copy.deepcopy(messages)
    final = messages[-1].get("content")
    if final is None:
        raise ValueError('continue_final_message is set but the final message has no "content" to continue!')
    if "content" not in template:
        raise ValueError('continue_final_message is set to "content" but this is not an accepted field in the chat_template')
    if isinstance(final, (list, tuple)):
        for block in reversed(final):
            if "text" in block:
                final = block["text"]
                block["text"] = block["text"] + CONTINUE_FINAL_MESSAGE_TAG
                break
        else:
            raise ValueError(
                "continue_final_message is set but we could not find any text to continue in the final message!"
            )
    else:
        messages[-1]["content"] = final + CONTINUE_FINAL_MESSAGE_TAG
    return messages, final


def cut_at_final_message(rendered, final):
    """Truncate a render at the continuation tag, as Transformers 5.x does."""
    if final.strip() not in rendered or CONTINUE_FINAL_MESSAGE_TAG.strip() not in rendered:
        raise ValueError(
            "continue_final_message is set but the final message does not appear in the chat "
            "after applying the chat template"
        )
    location = rendered.rindex(CONTINUE_FINAL_MESSAGE_TAG.strip())
    if rendered[location : location + len(CONTINUE_FINAL_MESSAGE_TAG)] == CONTINUE_FINAL_MESSAGE_TAG:
        return rendered[:location]
    # The template trimmed trailing spacing after the tag.
    return rendered[:location].rstrip()


# GGUF header value types: scalar struct formats, 8 = string, 9 = array.
_GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
                 10: "Q", 11: "q", 12: "d"}


def read_gguf_metadata(path):
    """Read a GGUF file's key/value header without touching tensor data.

    llama.cpp's own metadata view omits every array, yet some hyperparameters
    (e.g. Gemma 4's per-layer KV head counts) are stored only as arrays. String
    arrays (vocabulary, merges) are skipped; numeric arrays become lists.
    """
    import struct

    with open(path, "rb") as handle:

        def read(fmt):
            fmt = "<" + fmt
            return struct.unpack(fmt, handle.read(struct.calcsize(fmt)))[0]

        def string():
            return handle.read(read("Q")).decode("utf-8", errors="replace")

        if handle.read(4) != b"GGUF" or read("I") not in (2, 3):
            raise ValueError(f"{path} is not a little-endian GGUF v2/v3 file")
        read("Q")  # tensor count
        values = {}
        for _ in range(read("Q")):
            key, kind = string(), read("I")
            if kind == 8:
                values[key] = string()
            elif kind == 9:
                item, count = read("I"), read("Q")
                if item == 8:
                    for _ in range(count):
                        handle.seek(read("Q"), 1)
                    continue
                if item not in _GGUF_SCALARS:
                    raise ValueError(f"Unsupported GGUF array type in {key!r}")
                fmt = "<%d%s" % (count, _GGUF_SCALARS[item])
                values[key] = list(struct.unpack(fmt, handle.read(struct.calcsize(fmt))))
            elif kind in _GGUF_SCALARS:
                values[key] = read(_GGUF_SCALARS[kind])
            else:
                raise ValueError(f"Unsupported GGUF value type in {key!r}")
        return values


# GGUF stores llama.cpp architecture names, not HF model_type strings. These are
# the text backbones named by hf_prompt_policies.KNOWN_PROFILES; one GGUF name may
# cover several HF text configs, which are then told apart by size alone.
GGUF_MODEL_TYPES = {
    "qwen35": ("qwen3_5_text",),
    "qwen35moe": ("qwen3_5_moe_text",),
    "gemma4": ("gemma4_text", "gemma4_unified_text"),
}


def gguf_backbone_config(metadata, vocab_size):
    """Translate GGUF hyperparameters into the HF text-config fingerprint fields.

    The result feeds hf_prompt_policies.resolve_prompt_policy, so a GGUF file
    receives the same architecture/size recommendation as its HF checkpoint,
    independent of file, repository, or served names. Conversion conventions:
    block_count includes appended MTP (nextn) layers; per-layer arrays describe
    sliding-window layers where HF's head_dim/num_key_value_heads do (Gemma 4
    stores the full-attention head size as key_length and the sliding one as
    key_length_swa); absent or zero expert fields mean a dense model.
    """
    arch = str(metadata.get("general.architecture", ""))

    def get(key):
        return metadata.get(f"{arch}.{key}")

    def first(value):
        return value[0] if isinstance(value, list) and value else value

    sliding = get("attention.sliding_window_pattern")
    layer = sliding.index(True) if isinstance(sliding, list) and True in sliding else 0
    kv_heads = get("attention.head_count_kv")
    if isinstance(kv_heads, list):
        kv_heads = kv_heads[layer] if layer < len(kv_heads) else first(kv_heads)
    layers = get("block_count")
    if layers is not None:
        layers -= get("nextn_predict_layers") or 0
    head_dim = get("attention.key_length_swa") or get("attention.key_length")
    text = {
        "hidden_size": get("embedding_length"),
        "num_hidden_layers": layers,
        "num_attention_heads": first(get("attention.head_count")),
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "intermediate_size": first(get("feed_forward_length")) or None,
        "num_experts": get("expert_count") or None,
        "moe_intermediate_size": get("expert_feed_forward_length") or None,
        "num_experts_per_tok": get("expert_used_count") or None,
        "vocab_size": vocab_size,
    }
    candidates = GGUF_MODEL_TYPES.get(arch, (f"gguf:{arch}",))
    for model_type in candidates:
        signature = tuple(
            model_type if field == "model_type"
            else text["num_experts_per_tok"] if field == "active_experts"
            else text[field]
            for field in PROFILE_FIELDS
        )
        if any(signature == expected for _, _, expected in KNOWN_PROFILES):
            break
    else:
        model_type = candidates[0]
    return {"model_type": model_type, "text_config": {"model_type": model_type, **text}}


# One question of each type. Loading compiles it with the selected policy and
# the GGUF's own template/tokenizer, so a template that cannot render the policy
# (e.g. no reasoning_content support) or unstable answer labels fail at startup
# rather than on every request.
STARTUP_PROBE_REQUEST = {
    "model": "startup-probe",
    "state": "A small cat sleeps on a red sofa.",
    "questions": {
        "choice": {"type": "choice", "instructions": "Which animal?",
                   "criteria": {"cat": "A cat", "dog": "A dog"}},
        "score": {"type": "score", "instructions": "Is an animal present?",
                  "criteria": ["absent", "present"]},
        "noul": {"type": "noul", "instructions": "Is the sofa red?"},
    },
}


def load_service(
    model_name,
    *,
    revision=None,
    gguf_file=None,
    prompt_policy=None,
    backend="llama-cpp",
    subfolder=None,
    rope_factor=1,
    device="auto",
    dtype="bfloat16",
    n_gpu_layers=None,
    max_model_len=16384,
    max_batch_size=32,
    max_batch_tokens=32768,
    max_request_branches=100,
    prefix_sharing="auto",
    chat_template_file=None,
    served_model_name=None,
    enforce_model_id=False,
    max_choice_options=255,
    mmproj=None,
    max_image_width=None,
    max_image_height=None,
    default_image_max_width=None,
    default_image_max_height=None,
):
    """Load a model and return a ready-to-use service, without starting HTTP.

    An omitted prompt_policy selects a known architecture/size recommendation
    from the GGUF header; unknown profiles warn and fall back to baseline.
    Explicit strings always win. device selects weight placement: cpu keeps
    everything on the host, auto offloads every layer to any available
    llama.cpp backend (Vulkan included) and falls back to CPU where no device
    exists. dtype selects the KV cache element type. max_model_len limits each
    complete compiled prompt and sizes the unified KV pool. max_batch_tokens
    limits packed suffix tokens per decode, not the shared-prefix prefill or
    total KV memory. max_request_branches caps questions admitted in a single
    request; max_choice_options separately caps options per Choice question.

    served_model_name is the public ID for discovery and responses (default:
    model_name); with enforce_model_id, requests must name it.
    chat_template_file replaces the GGUF's embedded tokenizer.chat_template,
    e.g. when a file was converted with an outdated template.
    mmproj names the model's GGUF vision projector (see resolve_mmproj_path)
    and enables image chat through llama.cpp's libmtmd. The image resize
    options bound decoded images before the projector's own preprocessing.

    The GGUF vocabulary, chat template, prompt policy, and Choice label capacity
    are validated from a vocabulary-only load before any weights are loaded:
    STARTUP_PROBE_REQUEST must compile under the selected policy.
    The loader sets service concurrency to one: separate requests are
    serialized, while branches within a request are batched. The backend's
    thread lock also prevents overlap if cancellation releases admission
    before a decode ends.
    """
    from hf_vision import validate_image_resize_config
    validate_image_resize_config(max_image_width, max_image_height,
                                 default_image_max_width, default_image_max_height)
    if backend == 'laya' and any(value is not None for value in (
            max_image_width, max_image_height, default_image_max_width, default_image_max_height)):
        raise ValueError('Image resize options require --backend llama-cpp')
    if backend == 'laya' and mmproj is not None:
        raise ValueError('--mmproj requires --backend llama-cpp; Laya is text-only')
    validate_rope_factor(rope_factor)
    if prompt_policy is not None:
        validate_policy(prompt_policy)
    if type(max_choice_options) is not int or not 2 <= max_choice_options <= 255:
        raise ValueError("max_choice_options must be between 2 and 255")
    if served_model_name is not None and not served_model_name.strip():
        raise ValueError("served_model_name must not be empty")
    public_model = served_model_name if served_model_name is not None else model_name
    if backend == "laya" and prompt_policy not in (None, "baseline"):
        raise ValueError("Prompt policies apply only to --backend llama-cpp; Laya uses native formatting")
    if backend == "laya":
        # Resolve the revision ourselves because the SDK does not expose it.
        # Import only when selected; a llama.cpp installation stays usable.
        try:
            import laya
        except ImportError as exc:
            raise ImportError(
                "Install Laya support with pip install -e './hf-server[laya]'"
            ) from exc
        path = model_name
        if not Path(path).is_dir():
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                model_name,
                revision=revision,
                allow_patterns=[f"{subfolder}/*"]
                if subfolder
                else [
                    "rl_agent_config.json",
                    "model.safetensors",
                    "encoder/*",
                    "tokenizer/*",
                ],
            )
        agent = laya.load(
            path, subfolder=subfolder, device=None if device == "auto" else device
        )
        extend_laya_rope(agent, rope_factor)
        return DecisionService(
            public_model,
            None,
            LayaBackend(agent, max_model_len),
            enforce_model_id=enforce_model_id,
            max_choice_options=max_choice_options,
            concurrency=1,
            max_request_branches=max_request_branches,
            metadata={
                "backend": "laya",
                "model_revision": revision,
                "subfolder": subfolder,
                "rope_factor": rope_factor,
                "native_sequence_limit": agent.cfg["max_len"],
            },
        )
    if backend != "llama-cpp":
        raise ValueError(f"Unknown backend: {backend}")
    if subfolder:
        raise ValueError("--subfolder is currently supported only with --backend laya")
    if dtype not in KV_CACHE_TYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if prefix_sharing not in ("auto", "on", "off"):
        raise ValueError(f"Unknown prefix sharing mode: {prefix_sharing}")
    resolved_layers = resolve_n_gpu_layers(device, n_gpu_layers)
    # Heavy dependencies are local to loading, so CLI help and source inspection
    # do not initialize a backend or import the model classes.
    import llama_cpp
    from llama_cpp import _internals

    chat_template = (
        Path(chat_template_file).read_text(encoding="utf-8")
        if chat_template_file
        else None
    )
    gguf_path = resolve_gguf_path(model_name, revision, gguf_file)
    # Resolve (and download) the projector before any weights load.
    mmproj_path = (
        resolve_mmproj_path(mmproj, model_name, gguf_path, revision)
        if mmproj is not None else None
    )
    # A vocabulary-only load reads the tokenizer and chat template without any
    # weights, so an unusable template, policy, or label capacity fails fast.
    vocab_params = llama_cpp.llama_model_default_params()
    vocab_params.vocab_only = True
    vocab = _internals.LlamaModel(
        path_model=gguf_path, params=vocab_params, verbose=False
    )
    try:
        prompt_policy, policy_selection = resolve_prompt_policy(
            gguf_backbone_config(read_gguf_metadata(gguf_path), vocab.n_vocab()),
            prompt_policy,
        )
        probe = PromptCompiler(
            LlamaCppTokenizer(vocab, chat_template=chat_template),
            max_tokens=2**31,
            prompt_policy=prompt_policy,
            max_choice_options=max_choice_options,
        )
        probe.validate_choice_capacity()
        try:
            probe.compile(STARTUP_PROBE_REQUEST)
        except Exception as exc:
            raise ValueError(
                f"This GGUF's chat template/tokenizer cannot serve prompt policy "
                f"{prompt_policy!r} ({type(exc).__name__}: {exc}). Pass "
                "--chat-template-file with a template that supports it, or choose "
                "another --classifier-prompt-policy (baseline works with any "
                "chat template)"
            ) from exc
    finally:
        vocab.close()
    model_params = llama_cpp.llama_model_default_params()
    # llama.cpp encodes "all layers" as INT32 max; -1 is this API's spelling.
    model_params.n_gpu_layers = (
        0x7FFFFFFF if resolved_layers == -1 else resolved_layers
    )
    model = _internals.LlamaModel(
        path_model=gguf_path, params=model_params, verbose=False
    )
    # Recurrent and hybrid architectures carry rolling state alongside (or
    # instead of) attention cells. Recent llama.cpp does duplicate that state in
    # llama_memory_seq_cp, but support varies by architecture and build, so these
    # models default to per-branch prefill: correct everywhere, and only as
    # expensive as the prefill it stops sharing. --prefix-sharing opts back in.
    stateful = bool(
        llama_cpp.llama_model_is_recurrent(model.model)
        or llama_cpp.llama_model_is_hybrid(model.model)
    )
    share_prefix = not stateful if prefix_sharing == "auto" else prefix_sharing == "on"
    n_ctx_train = model.n_ctx_train()
    if max_model_len > n_ctx_train:
        # The engine clamps its own n_ctx to the trained context; admission and
        # the compiler limit must agree with it or requests would be rejected
        # far from their validation point.
        max_model_len = n_ctx_train
    context_params = llama_cpp.llama_context_default_params()
    context_params.n_ctx = max_model_len
    # One decode must be able to hold the shared prefix and a full token budget.
    # llama.cpp additionally clamps n_batch to n_ctx on its own.
    context_params.n_batch = max(max_batch_tokens, max_model_len)
    context_params.n_seq_max = max_batch_size + 1
    # The unified KV pool is one shared array of n_ctx cells; sequence copies
    # with a partial position range require it (the split cache asserts).
    context_params.kv_unified = True
    context_params.type_k = KV_CACHE_TYPES[dtype]
    context_params.type_v = KV_CACHE_TYPES[dtype]
    configure_rope(context_params, rope_factor)
    try:
        context = _internals.LlamaContext(
            model=model, params=context_params, verbose=False
        )
    except BaseException:
        model.close()
        raise
    vision = None
    if mmproj_path is not None:
        from hf_vision import MtmdVision

        try:
            # The projector follows the text weights: GPU when any layer is offloaded.
            vision = MtmdVision(mmproj_path, model.model, use_gpu=resolved_layers != 0)
        except BaseException:
            context.close()
            model.close()
            raise
    # PromptCompiler's default comes from common.DEFAULT_TEMPLATE_VERSION.
    # Keep one compiler/backend pair for the service's loaded model/context.
    compiler = PromptCompiler(
        LlamaCppTokenizer(model, chat_template=chat_template),
        max_tokens=max_model_len,
        prompt_policy=prompt_policy,
        max_choice_options=max_choice_options,
        vision=vision,
        max_image_width=max_image_width,
        max_image_height=max_image_height,
        default_image_max_width=default_image_max_width,
        default_image_max_height=default_image_max_height,
    )
    compiler.validate_choice_capacity()
    backend = LlamaCppBackend(
        model,
        context,
        max_batch_size=max_batch_size,
        max_batch_tokens=max_batch_tokens,
        share_prefix=share_prefix,
        vision=vision,
    )
    return DecisionService(
        public_model,
        compiler,
        backend,
        enforce_model_id=enforce_model_id,
        max_choice_options=max_choice_options,
        concurrency=1,
        max_request_branches=max_request_branches,
        metadata={
            "backend": "llama-cpp",
            "prompt_policy": prompt_policy,
            "prompt_policy_selection": policy_selection,
            "image_input": vision is not None,
            "mmproj_path": mmproj_path,
            "max_image_width": max_image_width,
            "max_image_height": max_image_height,
            "default_image_max_width": default_image_max_width,
            "default_image_max_height": default_image_max_height,
            "model_revision": revision,
            "gguf_path": gguf_path,
            "chat_template_source": chat_template_file or "gguf",
            "n_gpu_layers": resolved_layers,
            "kv_cache_dtype": dtype,
            "stateful_architecture": stateful,
            "prefix_sharing": share_prefix,
            "rope_factor": rope_factor,
        },
    )


def main():
    """Parse process settings, load the service, then run its ASGI application.

    host/port are Uvicorn settings; the other arguments configure loading and
    inference. Model loading happens before the listener starts accepting work.
    --help exits during parsing and therefore does not load any weights.
    """
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--gguf-file",
        help="GGUF filename to download from a Hugging Face repository",
    )
    parser.add_argument("--served-model-name", help="Public model ID for discovery and responses (default: --model)")
    parser.add_argument("--enforce-model-id", action="store_true", help="Reject request model IDs other than the served name")
    parser.add_argument("--max-choice-options", type=int, default=255, help="Maximum Choice options, 2–255 (default: 255)")
    parser.add_argument(
        "--classifier-prompt-policy", dest="prompt_policy",
        choices=PROMPT_POLICIES, default=None,
        help="Explicit format override; omitted: match known architecture/size, otherwise warn and use baseline. Named policies require state",
    )
    parser.add_argument(
        "--backend", choices=["llama-cpp", "laya"], default="llama-cpp"
    )
    parser.add_argument(
        "--subfolder", help="Laya checkpoint subfolder, e.g. multilingual"
    )
    parser.add_argument(
        "--rope-factor",
        "--laya-rope-factor",
        dest="rope_factor",
        type=float,
        default=1,
        help="Experimental linear RoPE interpolation factor (default: 1, disabled)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16",
        help="KV cache element type; GGUF weights keep their own quantization",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=None,
        help="llama.cpp layer offload count; -1 for all, overrides --device",
    )
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument(
        "--mmproj",
        help="GGUF vision projector enabling image chat: a path, or a file name "
             "next to the model / in the --model Hugging Face repository",
    )
    parser.add_argument("--max-image-width", type=int, default=None,
                        help="Hard preprocessor image width cap; downscale preserving aspect ratio (default: unset)")
    parser.add_argument("--max-image-height", type=int, default=None,
                        help="Hard preprocessor image height cap; downscale preserving aspect ratio (default: unset)")
    parser.add_argument("--default-image-max-width", type=int, default=None,
                        help="Default image resize width, overridable per request within the hard cap (default: hard cap)")
    parser.add_argument("--default-image-max-height", type=int, default=None,
                        help="Default image resize height, overridable per request within the hard cap (default: hard cap)")
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--max-request-branches", type=int, default=100)
    parser.add_argument(
        "--prefix-sharing",
        choices=["auto", "on", "off"],
        default="auto",
        help="Share prefix KV cells across branches; auto disables it for "
             "recurrent/hybrid architectures",
    )
    parser.add_argument(
        "--chat-template-file",
        help="Jinja chat template replacing the GGUF's embedded tokenizer.chat_template",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = vars(parser.parse_args())
    host, port, model = args.pop("host"), args.pop("port"), args.pop("model")
    uvicorn.run(create_app(load_service(model, **args)), host=host, port=port)


if __name__ == "__main__":
    main()
