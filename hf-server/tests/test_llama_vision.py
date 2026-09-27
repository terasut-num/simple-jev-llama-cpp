"""Image chat through llama.cpp's libmtmd: compilation, orchestration, fidelity.

1. Offline tests (no weights): a fake projector with real hf_vision layout
   objects exercises marker rendering, per-image spans and placeholder IDs,
   label boundaries, the stubbed engine's prefix/row decode order with M-RoPE
   positions, once-per-request image encoding, loading and CLI wiring.

2. Real-engine tests (skipped unless SIMPLE_JEV_GGUF and SIMPLE_JEV_MMPROJ
   name a local GGUF and its vision projector): the compiler's per-image token
   assembly must equal mtmd_tokenize of the complete rendered prompt, and the
   backend's scores, in both prefill strategies, must match llama.cpp's own
   mtmd_helper_eval_chunks reference over each complete prompt. These establish
   execution equivalence, not image-understanding quality.
"""

import base64
import ctypes
import io
import os
import sys

import pytest

import llama_cpp  # (presence gates this module)

import hf_server
import hf_vision
from hf_server import CompiledRequest, LlamaCppBackend, PromptCompiler, position_at
from hf_vision import MediaChunk, MtmdVision, RequestImages
from test_llama_backend import install_stub

GGUF = os.environ.get("SIMPLE_JEV_GGUF")
MMPROJ = os.environ.get("SIMPLE_JEV_MMPROJ")
DEVICE = os.environ.get("SIMPLE_JEV_DEVICE", "cpu")
real_engine = pytest.mark.skipif(
    GGUF is None or MMPROJ is None,
    reason="Set SIMPLE_JEV_GGUF and SIMPLE_JEV_MMPROJ to a local GGUF and its mmproj",
)
MARKER = "<__media__>"


def data_url(color="red", size=(32, 32), shape=None):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, "white" if shape else color)
    if shape == "circle":
        width, height = size
        ImageDraw.Draw(image).ellipse(
            (width // 4, height // 4, 3 * width // 4, 3 * height // 4), fill=color
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def payload(first="red", second="blue"):
    return {
        "model": "test",
        "messages": [
            {"role": "system", "content": "Inspect the supplied pictures."},
            {"role": "user", "content": [
                {"type": "text", "text": "First picture:"},
                {"type": "image_url", "image_url": {"url": data_url(first)}},
            ]},
            {"role": "assistant", "content": "I will inspect it."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url(second, (48, 32))}},
                {"type": "text", "text": "Compare with this picture."},
            ]},
        ],
        "questions": {
            "color": {"type": "choice", "instructions": "First color?",
                      "criteria": {"red": None, "blue": None}},
            "red": {"type": "noul", "instructions": "Is the first picture red?"},
            "level": {"type": "score", "instructions": "How red is the first?",
                      "criteria": ["not red", "red"]},
        },
    }


class ByteTokenizer:
    """Byte IDs plus a role template; the media marker stays literal text."""

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf8"))

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True,
                          "enable_thinking": False}
        def text(content):
            return content if isinstance(content, str) else "".join(b["text"] for b in content)

        return "\n".join(f"{m['role']}: {text(m['content'])}" for m in messages) + "\nassistant: "


class FakeVision:
    """Projector double: real layout objects, recorded encodes and decodes.

    Every image becomes [begin token, n_tokens embeddings over n_pos M-RoPE
    positions, end token], mirroring mtmd's layout for Qwen-VL projectors.
    """

    marker = MARKER
    tokenize = MtmdVision.tokenize

    def __init__(self, n_tokens=6, n_pos=3):
        self.n_tokens, self.n_pos = n_tokens, n_pos
        self.prepared, self.encodes, self.decodes = [], [], []

    def prepare(self, images):
        self.prepared.append([image.size for image in images])
        items, placeholder = [], -1
        for index, _ in enumerate(images):
            chunk = MediaChunk(1000 + index, self.n_tokens, self.n_pos, placeholder)
            placeholder -= self.n_tokens
            items.append([[250], chunk, [251]])
        return RequestImages(items, [])

    def encode(self, media, chunk):
        if chunk not in media.embeddings:
            self.encodes.append(chunk.pointer)
            media.embeddings[chunk] = object()
        return media.embeddings[chunk]

    def decode(self, context, media, chunk, n_past, seq_id, n_batch):
        self.encode(media, chunk)
        self.decodes.append((seq_id, n_past, chunk.pointer))
        return n_past + chunk.n_pos


def compile_images(vision=None, **kwargs):
    vision = vision or FakeVision()
    compiler = PromptCompiler(ByteTokenizer(), max_choice_options=50, vision=vision, **kwargs)
    return compiler.compile(payload()), vision


# Offline compilation


def test_images_render_as_markers_and_compile_to_spans():
    request = payload()
    compiled, vision = compile_images()
    assert vision.prepared == [[(32, 32), (48, 32)]]  # Decoded/prepared once per request.
    first = compiled.branches[0]
    # Image blocks became media-marker text blocks; turn order is unchanged.
    users = [m["content"] for m in first.messages if m["role"] == "user"]
    assert users[0] == [{"type": "text", "text": "First picture:"}, {"type": "text", "text": MARKER}]
    assert users[1] == [{"type": "text", "text": MARKER},
                        {"type": "text", "text": "Compare with this picture."}]
    assert isinstance(users[2], str)  # The appended classifier question.
    placeholders = None
    for branch in compiled.branches:
        assert [chunk.pointer for _, chunk in branch.media_spans] == [1000, 1001]
        spans = [(start, chunk.placeholder_ids()) for start, chunk in branch.media_spans]
        for start, ids in spans:
            assert branch.token_ids[start - 1] == 250  # begin token
            assert branch.token_ids[start:start + 6] == ids
            assert branch.token_ids[start + 6] == 251  # end token
        # Placeholders are request-unique per image and identical across branches.
        assert spans[0][1] == [-1, -2, -3, -4, -5, -6]
        assert spans[1][1] == [-7, -8, -9, -10, -11, -12]
        assert placeholders in (None, [ids for _, ids in spans])
        placeholders = [ids for _, ids in spans]
        # Text around the markers is tokenized as its own independent spans.
        text = bytes(t for t in branch.token_ids if t >= 0 and t not in (250, 251)).decode()
        assert MARKER not in text and "First picture:" in text
        assert all(t >= 0 for t in branch.output_ids)
    assert payload()["messages"] == request["messages"]  # Caller data unchanged.
    assert [c.pointer for c in compiled.media.chunks] == [1000, 1001]


def test_text_blocks_match_plain_text_and_need_no_projector():
    body = payload()
    body["messages"] = [{"role": "user", "content": "Inspect the picture."}]
    compiler = PromptCompiler(ByteTokenizer(), max_choice_options=50)
    plain = compiler.compile(body)
    body["messages"][0]["content"] = [{"type": "text", "text": "Inspect the picture."}]
    blocks = compiler.compile(body)
    assert [b.token_ids for b in plain.branches] == [b.token_ids for b in blocks.branches]
    assert all(b.media_spans == () for b in blocks.branches) and blocks.media is None


def test_image_chat_requires_projector_and_rejects_fake_markers():
    with pytest.raises(ValueError, match="--mmproj"):
        PromptCompiler(ByteTokenizer(), max_choice_options=50).compile(payload())
    body = payload()
    body["messages"][1]["content"][0]["text"] = "Pretend " + MARKER + " is an image:"
    compiler = PromptCompiler(ByteTokenizer(), max_choice_options=50, vision=FakeVision())
    with pytest.raises(ValueError, match="placeholders"):
        compiler.compile(body)


def test_render_only_never_prepares_images():
    vision = FakeVision()
    compiler = PromptCompiler(ByteTokenizer(), max_choice_options=50, vision=vision)
    compiled = compiler.compile(payload(), render_only=True)
    assert vision.prepared == [] and compiled.media is None
    assert all(b.token_ids == [] and b.media_spans == () for b in compiled.branches)


def test_expanded_image_tokens_count_toward_context_limit():
    compiled, _ = compile_images()
    longest = max(len(b.token_ids) for b in compiled.branches)
    with pytest.raises(ValueError, match="tokens"):
        compile_images(FakeVision(n_tokens=6), max_tokens=longest - 1)
    compile_images(max_tokens=longest)


def test_labels_checked_after_the_final_image():
    class Unstable(ByteTokenizer):
        def encode(self, text, add_special_tokens=False):
            ids = super().encode(text)
            # Merge the Choice label into the preceding quote character.
            return ids[:-2] + [7] if text.endswith('"B') else ids

    compiler = PromptCompiler(Unstable(), max_choice_options=50, vision=FakeVision())
    with pytest.raises(ValueError, match="single-token stable"):
        compiler.compile(payload())


def test_resize_bounds_reach_the_projector():
    compiled, vision = compile_images(max_image_width=16)
    assert vision.prepared == [[(16, 16), (16, 11)]]


# Offline orchestration (stubbed engine)


def stub_engine(monkeypatch, n_ctx=8192):
    context, calls = install_stub(monkeypatch, n_ctx=n_ctx)
    sys.modules["llama_cpp"].llama_n_batch = lambda ctx: 4096
    return context, calls


async def test_images_in_shared_prefix_decode_once_on_sequence_zero(monkeypatch):
    compiled, vision = compile_images()
    context, calls = stub_engine(monkeypatch)
    result = await LlamaCppBackend(None, context, vision=vision).score(compiled)
    metrics = result.metrics
    # Both images precede every question, so they sit in the shared prefix.
    assert metrics["prefill_strategy"] == "multimodal_shared_prefix"
    assert vision.encodes == [1000, 1001] and metrics["vision_forwards"] == 2
    assert [(seq, pointer) for seq, _, pointer in vision.decodes] == [(0, 1000), (0, 1001)]
    assert metrics["image_decodes"] == 2
    branch = compiled.branches[0]
    (s0, c0), (s1, c1) = branch.media_spans
    # M-RoPE: each image advances 3 positions for its 6 embeddings.
    assert vision.decodes[0][1] == s0 and vision.decodes[1][1] == s1 - 3
    prefix_len = metrics["prefix_tokens"]
    prefix_pos = position_at(branch.media_spans, prefix_len)
    assert prefix_pos == prefix_len - 6
    text_decodes = [c[1] for c in calls if c[0] == "decode"]
    # Prefix text runs: before image 0, between images, after image 1.
    assert [rows[0][1] for rows in text_decodes[:3]] == [0, s0 + 3, s1]
    assert all(seq == 0 for rows in text_decodes[:3] for seq, *_ in rows)
    copies = [c for c in calls if c[0] == "seq_cp"]
    assert copies == [("seq_cp", 0, row + 1, 0, prefix_pos) for row in range(3)]
    # One packed suffix decode continues at each row's M-RoPE position.
    suffix = text_decodes[3]
    starts = {}
    for seq, pos, _, _ in suffix:
        starts.setdefault(seq, pos)
    assert set(starts.values()) == {prefix_pos}
    assert set(result.logits) == {b.branch_id for b in compiled.branches}


async def test_per_branch_prefill_encodes_once_and_decodes_images_per_row(monkeypatch):
    compiled, vision = compile_images()
    context, calls = stub_engine(monkeypatch)
    result = await LlamaCppBackend(None, context, vision=vision, share_prefix=False).score(compiled)
    metrics = result.metrics
    assert metrics["prefill_strategy"] == "multimodal_independent"
    assert metrics["prefix_tokens"] == 0
    assert vision.encodes == [1000, 1001] and metrics["vision_forwards"] == 2
    assert sorted(seq for seq, _, _ in vision.decodes) == [1, 1, 2, 2, 3, 3]
    assert metrics["image_decodes"] == 6
    assert not [c for c in calls if c[0] == "seq_cp"]
    by_length = sorted(range(3), key=lambda i: len(compiled.branches[i].token_ids), reverse=True)
    packed = [c[1] for c in calls if c[0] == "decode"][-1]
    for row, i in enumerate(by_length):
        branch = compiled.branches[i]
        (_, c1) = branch.media_spans[1]
        split = branch.media_spans[1][0] + c1.n_tokens
        tokens = [token for seq, _, token, _ in packed if seq == row + 1]
        positions = [pos for seq, pos, _, _ in packed if seq == row + 1]
        assert tokens == branch.token_ids[split:]
        assert positions[0] == position_at(branch.media_spans, split) == split - 6
    assert set(result.logits) == {b.branch_id for b in compiled.branches}


async def test_image_branches_require_projector_and_cancel(monkeypatch):
    import asyncio
    import threading

    compiled, vision = compile_images()
    context, _ = stub_engine(monkeypatch)
    with pytest.raises(ValueError, match="projector"):
        await LlamaCppBackend(None, context).score(compiled)
    stop = threading.Event()
    instance = LlamaCppBackend(None, context, vision=vision, share_prefix=False)

    def cancel_after_first(*args):
        stop.set()
        return FakeVision.decode(vision, *args)

    vision.decode = cancel_after_first
    with pytest.raises(asyncio.CancelledError):
        instance._score(compiled, stop)
    # Stopped before the next row's image work and before any scoring decode.
    assert {seq for seq, _, _ in vision.decodes} == {1}


async def test_http_image_requests_without_projector_are_422():
    import httpx

    from hf_server import DecisionService, create_app

    service = DecisionService("m", PromptCompiler(ByteTokenizer(), max_choice_options=50), None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)),
                                 base_url="http://t") as client:
        for route in ["/v1/classifier", "/v1/systemone"]:
            response = await client.post(route, json=payload())
            assert response.status_code == 422 and "--mmproj" in response.text


# Loading and CLI


def test_loader_binds_projector_to_compiler_and_backend(monkeypatch, tmp_path):
    from conftest import fake_gguf_loader
    from test_prompt_policies import NativeTokenizer

    created = fake_gguf_loader(monkeypatch, NativeTokenizer())
    projector = tmp_path / "mmproj-F16.gguf"
    projector.write_bytes(b"GGUF")
    loads = []

    class Recorder:
        marker = MARKER

        def __init__(self, path, model, *, use_gpu):
            loads.append((path, model, use_gpu))

    monkeypatch.setattr(hf_vision, "MtmdVision", Recorder)
    service = hf_server.load_service("model.gguf", device="cpu", mmproj=str(projector),
                                     max_image_width=512, max_choice_options=50)
    weights = [m for m in created.models if not m.vocab_only]
    assert loads == [(str(projector), weights[0], False)]
    assert isinstance(service.compiler.vision, Recorder)
    assert service.backend.vision is service.compiler.vision
    assert service.compiler.max_image_width == 512
    assert service.metadata["image_input"] is True
    assert service.metadata["mmproj_path"] == str(projector)
    text_only = hf_server.load_service("model.gguf", device="cpu", max_choice_options=50)
    assert text_only.compiler.vision is None and text_only.metadata["image_input"] is False
    with pytest.raises(ValueError, match="Laya"):
        hf_server.load_service("x", backend="laya", mmproj=str(projector))


def test_projector_resolution_and_model_discovery(tmp_path, monkeypatch):
    from hf_server import resolve_gguf_path, resolve_mmproj_path

    folder = tmp_path / "repo"
    folder.mkdir()
    (folder / "Model-Q4_K_M.gguf").write_bytes(b"GGUF")
    (folder / "mmproj-F16.gguf").write_bytes(b"GGUF")
    # A projector beside the weights is not a second model candidate.
    assert resolve_gguf_path(str(folder), None) == str(folder / "Model-Q4_K_M.gguf")
    gguf = str(folder / "Model-Q4_K_M.gguf")
    assert resolve_mmproj_path("mmproj-F16.gguf", str(folder), gguf, None) == str(folder / "mmproj-F16.gguf")
    assert resolve_mmproj_path(str(folder / "mmproj-F16.gguf"), "any", gguf, None) == str(folder / "mmproj-F16.gguf")
    with pytest.raises(ValueError, match="not found"):
        resolve_mmproj_path("mmproj-F32.gguf", str(folder), gguf, None)
    with pytest.raises(ValueError, match=".gguf"):
        resolve_mmproj_path("../outside.bin", str(folder), gguf, None)
    downloads = []
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda **kw: downloads.append(kw) or "/cache/mmproj-F16.gguf")
    assert resolve_mmproj_path("mmproj-F16.gguf", "org/Model-GGUF", "/cache/model.gguf", "rev") == "/cache/mmproj-F16.gguf"
    assert downloads == [{"repo_id": "org/Model-GGUF", "filename": "mmproj-F16.gguf", "revision": "rev"}]


def test_cli_passes_mmproj(monkeypatch):
    import uvicorn

    captured = {}
    monkeypatch.setattr(hf_server, "load_service", lambda model, **kw: captured.update(kw) or object())
    monkeypatch.setattr(hf_server, "create_app", lambda service: service)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)
    monkeypatch.setattr(sys, "argv", ["simple-jev", "--model", "m", "--mmproj", "mmproj-F16.gguf"])
    hf_server.main()
    assert captured["mmproj"] == "mmproj-F16.gguf"


# Real engine


def real_pieces(n_ctx=4096):
    import llama_cpp
    from llama_cpp import _internals

    llama_cpp.llama_backend_init()
    params = llama_cpp.llama_model_default_params()
    params.n_gpu_layers = 0 if DEVICE == "cpu" else 0x7FFFFFFF
    model = _internals.LlamaModel(path_model=GGUF, params=params, verbose=False)
    context_params = llama_cpp.llama_context_default_params()
    context_params.n_ctx = n_ctx
    context_params.n_batch = n_ctx
    context_params.n_seq_max = 4
    context_params.kv_unified = True
    if DEVICE == "cpu":
        context_params.op_offload = False
    context = _internals.LlamaContext(model=model, params=context_params, verbose=False)
    vision = MtmdVision(MMPROJ, model.model, use_gpu=DEVICE != "cpu")
    stateful = bool(llama_cpp.llama_model_is_recurrent(model.model)
                    or llama_cpp.llama_model_is_hybrid(model.model))
    return model, context, vision, stateful


def mtmd_reference(vision, context, text, images, output_ids):
    """Tokenize the complete prompt natively and evaluate it on a fresh pool."""
    import llama_cpp
    from llama_cpp import mtmd_cpp

    chunks = mtmd_cpp.mtmd_input_chunks_init()
    bitmaps = []
    try:
        for image in images:
            data = image.tobytes()
            buffer = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
            bitmaps.append(mtmd_cpp.mtmd_bitmap_init(image.width, image.height, buffer))
        encoded = text.encode()
        prompt = mtmd_cpp.mtmd_input_text()
        prompt.text, prompt.text_len = encoded, len(encoded)
        prompt.add_special, prompt.parse_special = False, True
        array = (mtmd_cpp.mtmd_bitmap_p_ctypes * len(bitmaps))(*bitmaps)
        assert mtmd_cpp.mtmd_tokenize(vision.ctx, chunks, ctypes.byref(prompt), array, len(bitmaps)) == 0
        layout = []
        for index in range(mtmd_cpp.mtmd_input_chunks_size(chunks)):
            chunk = mtmd_cpp.mtmd_input_chunks_get(chunks, index)
            if mtmd_cpp.mtmd_input_chunk_get_type(chunk) == mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_TEXT:
                count = ctypes.c_size_t()
                tokens = mtmd_cpp.mtmd_input_chunk_get_tokens_text(chunk, ctypes.byref(count))
                layout.extend(int(tokens[k]) for k in range(count.value))
            else:
                layout.append(("image", int(mtmd_cpp.mtmd_input_chunk_get_n_tokens(chunk)),
                               int(mtmd_cpp.mtmd_input_chunk_get_n_pos(chunk))))
        llama_cpp.llama_memory_clear(context.memory, True)
        n_past = llama_cpp.llama_pos(0)
        assert mtmd_cpp.mtmd_helper_eval_chunks(
            vision.ctx, context.ctx, chunks, 0, 0, llama_cpp.llama_n_batch(context.ctx),
            True, ctypes.byref(n_past)) == 0
        logits = llama_cpp.llama_get_logits_ith(context.ctx, -1)
        return layout, [float(logits[i]) for i in output_ids]
    finally:
        for bitmap in bitmaps:
            mtmd_cpp.mtmd_bitmap_free(bitmap)
        mtmd_cpp.mtmd_input_chunks_free(chunks)


def softmax(values):
    import math

    top = max(values)
    weights = [math.exp(v - top) for v in values]
    return [w / sum(weights) for w in weights]


# llama.cpp's CPU kernels are not batch-invariant: identical tokens decoded in a
# differently sized or composed batch (a 1-token tail, several packed rows)
# round differently, for text-only requests too. Measured on CPU: up to ~0.6
# logits for Qwen3-VL-2B-Q8_0 at ~500-token branches, while single rows with an
# identical decode partition match exactly. Packed requests are therefore bounded
# in label-probability space, like upstream's documented ~0.20 BF16 bound.
PACKED_PROBABILITY_TOLERANCE = 0.2


@real_engine
async def test_image_scores_match_mtmd_reference():
    """Per-image assembly equals mtmd_tokenize; decodes equal mtmd's own evaluation.

    Strict (1e-3 logits on CPU): each branch scored alone in the per-branch
    strategy has the same decode partition as mtmd_helper_eval_chunks over its
    complete prompt, so token assembly, once-per-request image encoding,
    M-RoPE image and text positions must reproduce it. The shared-prefix
    strategy for one branch must equal a single-sequence decode with the same
    prefix/suffix split, proving the image-bearing prefix survives seq_cp.
    Bounded: full packed requests in both strategies, see above.
    """
    from dataclasses import replace

    from hf_server import LlamaCppTokenizer

    model, context, vision, _ = real_pieces()
    try:
        compiler = PromptCompiler(LlamaCppTokenizer(model), max_tokens=4096,
                                  max_choice_options=50, vision=vision)
        texts = []
        original = vision.tokenize

        def record(tokenizer, text, media):
            texts.append(text)
            return original(tokenizer, text, media)

        vision.tokenize = record
        body = payload()
        compiled = compiler.compile(body)
        _, images = hf_vision.image_messages(body["messages"])
        labels = {q.branch_id: q.output_labels for q in compiled.plan.questions}
        strict = 1e-3 if DEVICE == "cpu" else 0.15

        async def score(rows, share_prefix):
            backend = LlamaCppBackend(model, context, share_prefix=share_prefix, vision=vision)
            result = await backend.score(replace(compiled, branches=rows))
            return result, {b.branch_id: [result.logits[b.branch_id][label]
                                          for label in labels[b.branch_id]] for b in rows}

        references = {}
        for branch, text in zip(compiled.branches, texts):
            layout, reference = mtmd_reference(vision, context, text, images, branch.output_ids)
            references[branch.branch_id] = reference
            ours, index, spans = [], 0, dict(branch.media_spans)
            while index < len(branch.token_ids):
                if index in spans:
                    chunk = spans[index]
                    ours.append(("image", chunk.n_tokens, chunk.n_pos))
                    index += chunk.n_tokens
                else:
                    ours.append(branch.token_ids[index])
                    index += 1
            assert ours == layout
            result, got = await score([branch], False)
            assert result.metrics["vision_forwards"] == len(compiled.media.chunks)
            worst = max(abs(a - b) for a, b in zip(got[branch.branch_id], reference))
            assert worst < strict, f"per-branch image scores diverged by {worst}"

            backend = LlamaCppBackend(model, context, vision=vision)
            cut = len(branch.token_ids) - 1
            llama_cpp.llama_memory_clear(context.memory, True)
            backend._decode_range(llama_cpp, compiled, branch, 0, cut, 0)
            flagged = backend._decode_rows(llama_cpp, [(0, position_at(branch.media_spans, cut),
                                                        branch.token_ids[cut:], True)])
            logits = llama_cpp.llama_get_logits_ith(context.ctx, flagged[0])
            split_reference = [float(logits[i]) for i in branch.output_ids]
            result, got = await score([branch], True)
            assert result.metrics["prefill_strategy"] == "multimodal_shared_prefix"
            assert result.metrics["prefix_tokens"] == cut
            worst = max(abs(a - b) for a, b in zip(got[branch.branch_id], split_reference))
            assert worst < strict, f"shared-prefix image scores diverged by {worst}"

        for share_prefix in (True, False):
            result, got = await score(list(compiled.branches), share_prefix)
            assert result.metrics["prefill_strategy"] == (
                "multimodal_shared_prefix" if share_prefix else "multimodal_independent")
            assert result.metrics["vision_forwards"] == len(compiled.media.chunks)
            for branch_id, reference in references.items():
                worst = max(abs(a - b) for a, b in zip(softmax(got[branch_id]), softmax(reference)))
                assert worst < PACKED_PROBABILITY_TOLERANCE, (share_prefix, branch_id, worst)
    finally:
        vision.close()
        context.close()
        model.close()


@real_engine
async def test_image_service_end_to_end_http():
    import httpx

    from hf_server import create_app, load_service, unique_prompt_tokens

    service = load_service(GGUF, device=DEVICE, dtype="float32", mmproj=MMPROJ,
                           max_model_len=4096, max_choice_options=50, prompt_policy="baseline")
    body = {
        "model": "any",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Look at this picture."},
            {"type": "image_url", "image_url": {"url": data_url("red", (256, 192), "circle")}},
        ]}],
        "questions": {
            "color": {"type": "choice", "instructions": "What color is the shape?",
                      "criteria": {"red": None, "blue": None, "green": None}},
            "shape": {"type": "choice", "instructions": "What shape is shown?",
                      "criteria": {"circle": None, "square": None, "triangle": None}},
        },
    }
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)),
                                     base_url="http://t") as client:
            for route in ["/v1/classifier", "/v1/systemone"]:
                response = await client.post(route, json=body)
                assert response.status_code == 200, response.text
                answers = response.json()["answers"]
                assert answers["color"]["choice"] == "red"
                assert answers["shape"]["choice"] == "circle"
            compiled = service.compiler.compile(body)
            usage = response.json()["usage"]
            assert usage == {"input_tokens": unique_prompt_tokens(
                [b.token_ids for b in compiled.branches]), "output_tokens": 0}
            # Expanded image embeddings count as input tokens.
            assert usage["input_tokens"] > sum(c.n_tokens for c in compiled.media.chunks)
            bad = dict(body, media_io_kwargs={"video": {"fps": 1}})
            assert (await client.post("/v1/classifier", json=bad)).status_code == 422
    finally:
        service.compiler.vision.close()
        service.backend.context.close()
        service.backend.model.close()
