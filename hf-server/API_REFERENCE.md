# Simple-JEV HTTP API reference

This reference describes the standalone llama.cpp (GGUF) implementation in
`hf_server.py`, version 0.1.0. It does not require vLLM, PyTorch, or Transformers.
The API evaluates many
questions against one context and returns JSON in one non-streaming response.
It reads selected next-token logits; it does not generate prose answers.

## Endpoints and transport

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/v1/classifier` | Score the supplied questions. |
| POST | `/v1/systemone` | Exact alias of `/v1/classifier`; omitted from generated OpenAPI. |
| GET | `/v1/models` | OpenAI-style `object: "list"` with the served model ID and `x_max_choice_options`. No inference. |
| GET | `/health` | Returns `{"status":"ready","model":"<served model>"}` after service initialization. This does not run an inference probe. |
| GET | `/docs` | Interactive Swagger documentation. |
| GET | `/redoc` | Generated ReDoc documentation. |
| GET | `/openapi.json` | Generated request schema and route definitions. |

Send `Content-Type: application/json`. There are no classifier query parameters
or required custom headers. This standalone server implements no authentication,
API-key management or TLS itself. It has no chat-completions endpoint; `messages`
is an alternative way to supply classifier context.

The generated OpenAPI covers request schemas, but the handwritten response,
error, ignored-field and runtime-limit details below are more complete.

## Quick start

After installation, start the server with a GGUF model that carries a chat
template and single-token answer labels:

```bash
simple-jev --model Qwen/Qwen2.5-0.5B-Instruct-GGUF --host 0.0.0.0 --port 8000
```

Request model IDs are accepted without name checking by default; they never select or load a model. `--served-model-name` controls the ID in discovery, health and responses (default: `--model`). Add `--enforce-model-id` to require that served ID. For example, `--served-model-name jev-latest --enforce-model-id` enables strict SDK-name matching.

Example request:

```bash
curl --fail-with-body http://localhost:8000/v1/classifier \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "model": "Qwen/Qwen3.5-2B",
  "state": "Mia owns a red bicycle. Her dog is named Max.",
  "questions": {
    "color": {
      "type": "choice",
      "instructions": "What color is Mia's bicycle?",
      "criteria": {"red": null, "blue": null}
    },
    "support": {
      "type": "score",
      "instructions": "How strongly does the context support that Mia owns a bicycle?",
      "criteria": ["Unsupported", "Partially supported", "Fully supported"]
    },
    "dog": {
      "type": "noul",
      "instructions": "Is Mia's dog named Max?"
    }
  }
}
JSON
```

Illustrative response (probabilities and usage are examples, not measured output):

```json
{
  "model": "Qwen/Qwen3.5-2B",
  "answers": {
    "color": {
      "type": "choice",
      "choice": "red",
      "confidence": 0.97,
      "probabilities": {"red": 0.97, "blue": 0.03}
    },
    "support": {
      "type": "score",
      "score": 1.9,
      "confidence": 0.92,
      "probabilities": {"0": 0.02, "1": 0.06, "2": 0.92},
      "legend": {"0": "Unsupported", "1": "Partially supported", "2": "Fully supported"}
    },
    "dog": {"type": "noul", "noul": 0.96}
  },
  "usage": {"input_tokens": 600, "output_tokens": 0}
}
```

## Request body

Unknown top-level fields are ignored. Unknown question/option fields are rejected. Field names are case-sensitive. Examples use JSON strings and
numbers; clients should send the documented types rather than rely on Pydantic
coercion.

| Field | Type | Required/default | Behavior |
| --- | --- | --- | --- |
| `model` | string | Required; nonempty | Any nonempty ID is accepted by default. With `--enforce-model-id`, it must match the served name. The HTTP request does not load or switch models. |
| `state` | string, object, array, or null | Supply exactly one non-null `state` or `messages` | Shared context. Objects/arrays are serialized into prompt text; they are not executable state. A top-level number or boolean is not supported. |
| `messages` | array of messages or null | Alternative to `state`; at least one message | Text/image chat history rendered with the model's native template; see [image support](VISION.md). |
| `questions` | object mapping IDs to questions | Required; 1–256 entries at schema level | IDs must be nonempty strings. The server's branch limit is additionally enforced, default 100. |
| `options` | object | Defaults shown below | Response diagnostics; prompt/scoring rules are fixed by v1. |
| `tools` | array of objects or null | Omitted/null | Reserved in the schema; nonempty values are rejected by this implementation. |
| `mm_processor_kwargs` | object or null | Omitted/null | Reserved; nonempty values are rejected. |
| `media_io_kwargs` | object of objects or null | Omitted/null | llama.cpp: `{"image":{"max_width":1024,"max_height":768}}` resizes images to fit, preserving aspect ratio. Positive integers override server defaults, clamped to server hard caps. Other keys are rejected; Laya rejects nonempty values. |

Empty reserved containers are accepted but have no effect. Omit them normally.
Explicit `null` does not count as supplied context. An empty string or empty JSON
object/array does count as `state`. Supplying both non-null context fields, or
neither, returns 422.

### Chat messages

For this implementation each message contains only:

| Field | Supported value |
| --- | --- |
| `role` | `system`, `developer`, `user`, or `assistant` |
| `content` | String (including empty), or text/image_url blocks; images in user turns only |

Text blocks use `{"type":"text","text":"..."}`. Image blocks use
`{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`.
The URL may also be public HTTP(S), for example
`{"type":"image_url","image_url":{"url":"https://example.com/photo.jpg"}}`.
Image blocks need a server started with the model's GGUF vision projector
(`--mmproj`); without it they return 422. See [image support and limits](VISION.md)
for projector families, JPEG/WebP, multiple images, prefix reuse, and validation scope. Downloads enforce byte/time limits, standard
HTTP(S) ports, public-IP-only DNS pinning, TLS verification and redirect checks;
private-network URLs, URL userinfo and local paths are rejected. Ambient proxies
and cookies are never used.

`tool`/`function` roles, null content, audio/video blocks, tool calls, `name`, and
other extra message properties are rejected. A model's chat template may further
restrict roles or their order.

For chat input replace `state` in the example with:

```json
"messages": [
  {"role": "user", "content": "Mia owns a red bicycle."},
  {"role": "assistant", "content": "Her dog is named Max."}
]
```

Classifier instructions and question prompts are assembled around this context
using the default `shared_questions_deliberate` pattern and the tokenizer's
chat template. Clients cannot override that template or pattern through this API.

### Question fields

An **entry** below means a string, JSON object, JSON array, or null. Nested JSON
may contain ordinary JSON scalar values. Bare numbers and booleans are not valid
entries. `instructions` is required even though its value may be null.

| Question type | `instructions` | `criteria` |
| --- | --- | --- |
| `choice` | Required entry describing the question | Required object with 2–255 candidate IDs, subject to `--max-choice-options`, mapped to entries describing each candidate. Null descriptions are allowed. |
| `score` | Required entry describing what to evaluate | Required ordered array of 2–50 entries, lowest level first. |
| `noul` | Required entry describing a truth/yes-no proposition | Optional object with only `"true"` and/or `"false"` keys mapped to entries; default null. Neither key is required. |

Every question requires `type`, exactly `choice`, `score`, or `noul`. Unknown
question fields are rejected. Candidate IDs are the public choice values. Their
JSON insertion order determines label assignment and breaks exact ties; score
criteria retain array order. Unlike question IDs, empty choice candidate IDs are
not explicitly forbidden by the current schema.

Example truth rubric:

```json
{
  "type": "noul",
  "instructions": "Does the message request a refund?",
  "criteria": {
    "true": "The customer explicitly asks for money back.",
    "false": "The customer makes no refund request."
  }
}
```

### Options — all fields

```json
{"raw_logits": false}
```

`raw_logits` requests selected-token logits, visible only with advanced metrics.
The shared `v1` template fixes choice, score, and Noul behavior. `choice_mode`,
`score_mode`, and `score_format` are rejected. The server selects v1 at the prompt
builder boundary; HTTP request fields cannot override it.

There is no request temperature or sampling step. Softmax uses the selected
logits without temperature scaling. Probabilities are conditional on the
candidate/rating token set, not the entire vocabulary, and are not calibrated
probabilities of correctness.

## Scoring and response fields

Every successful response contains:

| Field | Meaning |
| --- | --- |
| `model` | Configured served model identifier, regardless of the request's model string. |
| `answers` | Object keyed by the supplied question IDs. |
| `usage.input_tokens` | Exact union of token prefixes across the compiled question branches. Shared prefixes count once. Includes classifier instructions, examples, template tokens and suffixes. |
| `usage.output_tokens` | Always 0 for this backend: it scores logits without sampling output tokens. |

Usage excludes padding. It is logical unique-prefix accounting, not a measurement
of all actual model work: suffix batches can recompute additional overlap beyond
the common seed prefix. It is not persistent-cache billing across requests.
GPU backends add kernel-level floating-point differences relative to CPU
scoring; see the precision notes in the README.

### Choice

For up to 50 options, one branch assigns the unchanged single-token labels `A`–`Z`, then `a`–`x`. For 51–255 options the entire question instead uses distinct, fixed-width two-letter uppercase labels selected deterministically for the tokenizer (e.g. `AA`, `AB`). Single-letter labels are not mixed with two-letter labels, so no label is a prefix or substring of another. Both forms use one branch for the
candidates. A softmax over candidate logits produces:

| Field | Meaning |
| --- | --- |
| `type` | `choice` |
| `choice` | Candidate ID with the greatest logit. Exact ties select the first candidate. |
| `confidence` | Winning candidate's softmax probability. |
| `probabilities` | Object mapping each candidate ID to its probability; sums approximately to 1. |

### Score

Uses one branch and labels `0`–`9` for up to 10 criteria; above 10 it uses letter
labels internally and maps them back to zero-based criterion indices.

| Field | Meaning |
| --- | --- |
| `type` | `score` |
| `score` | Expected zero-based criterion index: `sum(p[i] * i)`. May be fractional; range 0 to `N-1`. |
| `confidence` | Largest criterion probability, not a confidence interval for the expected score. |
| `probabilities` | Object keyed by numeric strings `"0"` through `"N-1"`, even when internal labels are letters. |
| `legend` | Object mapping those same numeric strings to the original criteria. |

### Noul

Noul always uses one nine-bin rating branch. Convert the expected rating to a
decimal `d` in [0.1, 0.9], then return:

```text
noul = clamp(0.01 + (d - 0.1) * 0.98 / 0.8, 0.01, 0.99)
```

Thus endpoint ratings 0.1 and 0.9 map to 0.01 and 0.99, with midpoint 0.5.
Default fields are `type: "noul"` and `noul`. There is no separate confidence
field. This is a transformed expected rating, not a binary-token softmax.

## Advanced metrics and raw logits

Start the server with `ENABLE_OPEN_JEV_ADVANCED_METRICS=1` to expose these fields.
Values `true`, `yes`, and `on` also enable it, case-insensitively. The environment
variable retains its original name after the Simple-JEV rename. It is read when
the service is constructed, not from each HTTP request.

### Additional answer fields

| Mode | Additional fields |
| --- | --- |
| Direct choice | `margin` (top two probability difference), `ties` (all equal-max-logit IDs), `calibrated: false`, `scoring: "direct_label_logits"`; `logits` keyed by candidate ID if requested. |
| Direct score | `variance` over criterion indices, `calibrated: false`, `scoring: "direct_level_logits"`, `score_mapping: "label_to_zero_based_level"`; `logits` keyed by numeric index strings if requested. |
| Noul | `rating`, `calibrated: false`. |

A Noul `rating` contains `bins` (nine ordered
values), `probabilities` (nine values), `expected_score`, `variance`, and `entropy`
(natural-log units). With `raw_logits: true` it also contains a nine-element
`logits` array. Variance follows the selected integer/decimal units.

### Top-level metadata

| Field | Value/meaning |
| --- | --- |
| `metadata.backend` | `llama-cpp` |
| `metadata.model_revision` | Startup `--revision`, or null. |
| `metadata.gguf_path` | Resolved local GGUF file path. |
| `metadata.chat_template_source` | `gguf` for the embedded template, otherwise the `--chat-template-file` path. |
| `metadata.prompt_policy` | Resolved startup prompt format. |
| `metadata.prompt_policy_selection` | `mode` (`explicit`, `architecture-size`, or `unknown-baseline`) plus the matched `profile` and GGUF-derived `signature` when auto-selected. |
| `metadata.n_gpu_layers` | Resolved layer offload count (`-1` = all layers). |
| `metadata.kv_cache_dtype` | KV cache element type selected by `--dtype`. |
| `metadata.stateful_architecture` | True when the model reports recurrent or hybrid state (`llama_model_is_recurrent` / `llama_model_is_hybrid`). Such models carry rolling state rather than per-position KV cells. |
| `metadata.prefix_sharing` | True when branches share prefix KV cells, false when each branch prefills its own complete prompt. Resolved from `--prefix-sharing` and `metadata.stateful_architecture`. |
| `metadata.rope_factor` | Startup `--rope-factor`. |
| `metadata.image_input` | True when a vision projector (`--mmproj`) is loaded and image chat is accepted. |
| `metadata.mmproj_path` | Resolved local projector GGUF path, or null. |
| `metadata.max_image_width`, `metadata.max_image_height`, `metadata.default_image_max_width`, `metadata.default_image_max_height` | Startup image resize bounds (null when unset). |
| `metadata.template_version` | The resolved template version: `v1` for baseline, `hf-<policy>-v1` for a named format. |
| `metadata.calibration` | `not_calibrated` |
| `metadata.usage_accounting` | `unique_token_prefixes_and_engine_leaf_outputs` (legacy identifier). |

### Top-level metrics

| Field | Meaning |
| --- | --- |
| `backend` | `llama-cpp` |
| `prefill_strategy` | Text: `shared_prefix`, or `per_branch` when prefix sharing is disabled and every branch prefills its own complete prompt. Images: `multimodal_shared_prefix` when every image span lies in the shared prefix, else `multimodal_independent` (each branch decodes the request's once-encoded image embeddings itself). |
| `prefix_tokens` | Length of the shared prefix actually decoded once, counting expanded image tokens. At least one token is left for each suffix, even for identical prompts; an image span is never split. `0` under `per_branch`. |
| `vision_forwards` | Image requests only: projector encodes, one per image chunk per request (a tiling projector can emit several chunks per image), never per question. |
| `image_decodes` | Image requests only: image-embedding decodes into the KV cache; one per image for a shared prefix, one per image per branch otherwise. |
| `suffix_batch_sizes` | Number of question/candidate branches in each packed suffix decode. |
| `engine_forwards` | Prefix decodes (one per text run and image span), per-row image-bearing decodes, plus packed suffix decodes. These are engine calls, not HTTP calls. |
| `branch_prompt_tokens` | Sum of all complete branch lengths, including repeated prefixes and expanded image tokens. |
| `computed_prompt_tokens` | Shared prefix length plus all packed suffix tokens decoded. |
| `logical_prefill_tokens` | Shared prefix length plus unpadded suffix lengths. |
| `padded_suffix_tokens` | Equal to the packed suffix tokens: rows are packed without padding tokens in this engine. |
| `branch_output_tokens` | 0 |
| `scored_positions` | Number of scoring branches. |
| `backend_seconds` | Backend elapsed time inside model lock. |
| `queue_seconds` | Wait for the service's request semaphore. |
| `total_seconds` | Service time through response construction, including queue, compilation and inference; excludes final network transmission. |

## Limits, batching and cancellation

The CLI service runs one model request at a time with up to 16 additional
requests waiting. Questions within a request execute in suffix batches. At
capacity (17 admitted requests), additional requests receive 429.

The default request limit is 100 branches, configurable with
`--max-request-branches`. Every question consumes exactly one branch. The schema
caps questions at 256. Choice supports 2–255 options, limited by `--max-choice-options` (default 255). Score remains limited to 50 levels.

Set all three independently at startup, for example:

```bash
simple-jev --model Qwen/Qwen3.8-27B \
  --max-request-branches 256 --max-model-len 32768 --max-choice-options 255
```

This permits up to 256 questions per request, each with up to 255 Choice options,
provided each rendered branch fits 32768 tokens. Three 255-option questions use
three branches, not 765. Setting a branch limit above 256 does not bypass the
schema cap. These limits are not per-request fields; `max_tokens` does not set
input length. Options, template overhead, and policy repetition count toward the
branch's input tokens. All maxima need not fit simultaneously. Input violations
return 422 rather than silently dropping questions, candidates, or context.

For limits above 50, the GGUF loader checks, from a vocabulary-only load before any weights are loaded, that the GGUF vocabulary has enough distinct, non-control, single-token two-letter labels after the rendered `{"answer": "` boundary. If not, startup fails with guidance to reduce the limit. Actual rendered answer boundaries are checked again per request; incompatible boundaries return 422. No candidates are truncated and multi-token scoring is not substituted. More options lengthen the prompt, so context/token limits still apply.

Each complete compiled branch, including shared context and appended question
instructions, must fit `--max-model-len`. There is no automatic truncation.
This CLI limit is not automatically clamped to the model's native context limit;
configure it appropriately for the model. Setting it higher does not add model
support for longer sequences.

The backend decodes the exact common prefix once on sequence 0 of a unified
KV pool, then copies the prefix cells into per-row sequences for each suffix
batch with `llama_memory_seq_cp`. Suffixes are sorted by length, batched under
`--max-batch-size`, the packed `--max-batch-tokens` budget, and the remaining KV
cells, then packed into one `llama_decode` call without padding tokens.
A single suffix larger than the token budget returns 422.
The prefix decode itself is not chunked by this budget. There is no persistent
cross-request prefix cache or continuous cross-request batching.

When prefix sharing is disabled — `--prefix-sharing off`, or `auto` on a
recurrent/hybrid architecture — there is no prefix decode and no `seq_cp` step.
Each branch decodes its complete prompt on its own sequence id, still packed
into one padding-free `llama_decode` per batch. Each row's length is then the
whole prompt rather than a suffix, which changes two limits: `--max-batch-tokens`
must be at least the longest compiled branch or the request returns 422, and
rows per decode are bounded by `n_ctx // longest_branch`. Sizing
`--max-model-len` and `--max-batch-tokens` equal to each other therefore scores
one branch per decode.

GPU backends (Vulkan) introduce kernel-level floating-point differences
relative to CPU; see the precision notes in the README. Use `--device cpu
--dtype float32` when exact numerical anchoring matters.

Client disconnects cancel the service task. An in-flight model forward cannot
be immediately interrupted; the backend observes cancellation between forwards
and keeps its model lock until safe to release.

## Errors

| Status | Meaning |
| --- | --- |
| 422 | Invalid JSON/schema, a model ID other than the served name under `--enforce-model-id`, invalid context combination, unsupported chat/media/tool input, branch/token limits, or another compiler/backend `ValueError`. |
| 429 | Request queue full; header `Retry-After: 1`, body `{"detail":"Scoring queue is full"}`. |
| 499 | Client disconnected, if a response can still be delivered: `{"detail":"Client disconnected"}`. |
| 500 | Unhandled runtime failure, such as a model execution error. No stable structured error body is guaranteed. |

Example semantic validation error:

```json
{
  "error": {
    "message": "Served model is 'qwen-gguf'",
    "type": "invalid_request_error",
    "code": 422,
    "param": null,
    "details": []
  }
}
```

Schema errors use the same envelope, with up to ten detail entries containing
`param`, `message`, and `type`. `param` identifies a dotted field path, with array
indices such as `[0]`; union type names can appear in paths. The top-level `param`
is the first detail's path. Extra errors are counted in the summary message.
Semantic errors may have no field path or details. Validation detail entries omit
submitted input values and exception contexts.

## Unknown top-level fields

All undeclared top-level fields are ignored, including completion settings and
custom client fields. For example, `stream: true` still returns ordinary JSON,
and `max_tokens` does not change the number of questions scored. Declared fields
remain validated; misspelled fields inside questions/options are rejected.

## Startup prompt policies

If the flag is omitted, known architecture/size profiles auto-select the
recommended policy; unknown profiles use baseline with a prominent tuning warning.
The profile is read from the GGUF header (see
[prompt format selection](README.md#prompt-format-selection)); no file or
model-name matching is used. Set `--classifier-prompt-policy baseline` to
preserve the former default or use plain-text `messages`. Advanced metadata
includes `prompt_policy` and `prompt_policy_selection` (mode/profile/signature).

An explicit `--classifier-prompt-policy` selects `baseline`, `examples_binary`,
`repeat_state`, `strict_mix_repeat2`, `shared_examples_binary`, `shared_repeat_state`,
or experimental `universal_shared`. It is a startup setting, not a request
field or header. Invalid names fail argument parsing; non-baseline policies
with `--backend laya` fail before loading weights. Before loading GGUF weights,
the server compiles a sample request with the resolved policy; a chat template or
vocabulary that cannot serve it stops startup with guidance to pass
`--chat-template-file` or choose another policy.

| Policy | Formatting | Noul |
|---|---|---|
| `baseline` | Explicit legacy format, including plain-text chat support | Nine bins mapped to [0.01,0.99] |
| `examples_binary` | Strict rules + worked examples; state once | Restricted probability of yes over no/yes, in [0,1] |
| `repeat_state` | Same as examples_binary; state repeated twice | Same binary probability |
| `strict_mix_repeat2` | Strict rules; full user-input block repeated twice | Original evaluated nine-bin wording and [0.01,0.99] mapping |
| `shared_examples_binary` | Uniform system/native thinking flag; type-specific instructions after shared context | Binary probability |
| `shared_repeat_state` | Same shared layout; state twice in shared prefix, chat once | Binary probability |
| `universal_shared` | Experimental universal rules and labelled catalogue before context; selector-only suffix | Binary probability |

Legacy `examples_binary`, `repeat_state`, and `strict_mix_repeat2` require `state`;
`messages` return 422. Baseline, shared_* and universal_shared support plain-text chat.
The chat-compatible policies also support [image chat](VISION.md) when a vision
projector is loaded. Choice branches in legacy/shared_*
policies use a fixed three-line native `[thinking]` prefill, not generated reasoning.
Baseline/universal_shared and Score/Noul branches do not use this prefill.
Default tuning excludes the three legacy policies. `--all-formats` in the tuning
tool explicitly includes them and the experimental universal format.
The chat template must accept text-block content and render assistant
`reasoning_content`, and the tokenizer must preserve the prefill and every
allowed single-token answer boundary. The prefill is rendered with Transformers'
`continue_final_message` rule, reproduced for llama.cpp's Jinja renderer. All policy content counts toward the complete branch token limit.

Binary Noul responses contain only `type` and `noul`, including with advanced
metrics; nine-bin rating diagnostics do not apply. Choice/Score response math
and usage accounting are unchanged. Advanced metadata reports a distinct
`hf-<policy>-v1` template version. Common v1 itself is not modified. These formats
do not change the model precision, inference backend, cache, batching, workers
or scheduler. See [model recommendations](README.md#prompt-format-selection).

## Server startup arguments — exhaustive list

These are process settings, not HTTP request fields. Both `simple-jev` and
`python -m hf_server` accept them.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--model` | Required | GGUF file path, directory containing exactly one `.gguf`, or a Hugging Face GGUF repository id. Default public model ID. |
| `--revision` | Unset | Hugging Face revision used when downloading the GGUF. |
| `--gguf-file` | Unset | One `.gguf` filename to download from a Hugging Face repository with multiple variants. |
| `--served-model-name` | value of `--model` | Public model ID in responses, health and discovery; does not change which GGUF is loaded. |
| `--mmproj` | Unset | GGUF vision projector enabling image chat through llama.cpp's libmtmd: a path, or a file name next to the resolved model GGUF / in the `--model` Hugging Face repository (downloaded on its own). Uses the GPU when any layer is offloaded. |
| `--max-image-width`, `--max-image-height` | Unset | Hard resize bounds on images passed to the vision projector. Requests above these are clamped, not rejected. |
| `--default-image-max-width`, `--default-image-max-height` | Corresponding hard cap | Default request resize bounds; overridable via `media_io_kwargs.image.max_width/max_height` within hard caps. Explicit defaults must be positive and not exceed hard caps. |
| `--enforce-model-id` | off | Reject request IDs other than the served name. |
| `--max-choice-options` | `255` | Choice cap from 2 to 255; Score/Noul unchanged. Above 50, the GGUF vocabulary must provide enough single-token two-letter labels, checked before weights load. |
| `--classifier-prompt-policy` | Omitted: architecture/size selection | Known GGUF header profiles auto-select a recommended format; unknown profiles warn and use baseline. Explicit values always override, including baseline. |
| `--chat-template-file` | Unset | Jinja chat template replacing the GGUF's embedded `tokenizer.chat_template`, for every format. |
| `--backend` | `llama-cpp` | `llama-cpp` or `laya`. |
| `--subfolder` | Unset | Laya checkpoint subfolder, e.g. `multilingual`. |
| `--rope-factor` | `1` | Experimental linear RoPE interpolation factor applied to the context. |
| `--device` | `auto` | Weight placement: `cpu` keeps everything on the host; `auto`/`gpu`/`vulkan` offload all layers to any available llama.cpp backend with CPU fallback. |
| `--dtype` | `bfloat16` | One of `float32`, `float16`, `bfloat16`: the KV cache element type. GGUF weights keep their own quantization. |
| `--n-gpu-layers` | Derived from `--device` | Explicit llama.cpp layer offload count; `-1` for all layers, overrides `--device`. |
| `--max-model-len` | `16384` | Maximum input tokens per complete rendered question branch, including context, instructions, options, template overhead, and repetition. Not generated output length or native context extension. Also sizes the unified KV pool and is clamped to the model's trained context. |
| `--max-batch-size` | `32` | Maximum suffix rows per decode; the context allows one prefix sequence plus this many row sequences. |
| `--max-batch-tokens` | `32768` | Maximum packed suffix tokens per decode; must be positive. Does not chunk or limit the prefix decode. Bounds complete branch lengths, not suffixes, when prefix sharing is disabled. |
| `--max-request-branches` | `100` | Maximum questions per request: each question uses one branch regardless of option count. Set 256 for the schema maximum; larger settings cannot bypass it. |
| `--prefix-sharing` | `auto` | `auto`, `on` or `off`. Whether branches share prefix KV cells via `llama_memory_seq_cp`. `auto` shares them for ordinary attention models and disables sharing for recurrent/hybrid architectures, which then prefill each branch separately. `on` forces sharing for every architecture; `off` forces per-branch prefill. |
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8000` | HTTP port. |
| `-h`, `--help` | — | Print argument help and exit. |

No CLI flags are currently provided for authentication, quantization, model
aliases, request queue size, or request concurrency. Image support requires
`--mmproj`, as documented in [VISION.md](VISION.md). The service requires GGUF
models with a chat template and suitable single-token rating/choice labels.
Recurrent and hybrid architectures, whose state cannot be assumed
sequence-copyable, are accepted and default to per-branch prefill; arbitrary
GGUF files are not guaranteed to work.

## Source of truth

- [Shared prompt structure](../common/PROMPT_STRUCTURE_V1.md)
- [Shared prompt builder](../common/prompt_builder.py)
- [Request schema](../common/request_schema.py)
- [GGUF server: chat rendering, inference, service, HTTP routes, and CLI](hf_server.py)
- [Scoring formulas](../common/response_scoring.py)
