# Simple-JEV

Standalone classifier HTTP API using llama.cpp (GGUF) through
`llama-cpp-python`, with Vulkan GPU acceleration. Classifier validation,
prompt text, and response scoring come from the sibling `common/` folder, and
the startup-selected prompt formats from `hf_prompt_policies.py`. Keep both
folders in the checkout. No Open-JEV, vLLM, PyTorch, or Transformers runtime is
required. The package includes `common` when installed/built from this repo.

## Install and run

Use Python 3.12 or newer. `llama-cpp-python` compiles llama.cpp from source, so
a C/C++ toolchain (and the Vulkan SDK for GPU acceleration) must be present:

```bash
cd /path/to/simple-jev
python -m venv .venv
source .venv/bin/activate
# Build with the Vulkan backend for GPU acceleration:
CMAKE_ARGS="-DGGML_VULKAN=ON" pip install llama-cpp-python --no-cache-dir
# Or, for CPU-only execution, leave CMAKE_ARGS unset.
pip install -e './hf-server[test]'
simple-jev --model /path/to/model.gguf --device auto \
  --dtype bfloat16 --max-model-len 32768 \
  --max-batch-size 32 --max-batch-tokens 32768 --port 8000
```

On Windows, set the same variable with PowerShell: `$env:CMAKE_ARGS="-DGGML_VULKAN=ON"`.

Or run the file directly from the checkout:

```bash
python hf-server/hf_server.py --model /path/to/model.gguf --device cpu
```

After installation, `python -m hf_server` accepts the same arguments.
`--model` accepts a local `.gguf` file path, a directory containing exactly
one `.gguf` file, or a Hugging Face repository id. Use `--gguf-file NAME.gguf`
to select one weight file from a repository with multiple GGUF variants.
`--chat-template-file PATH` replaces the GGUF's embedded chat template (see
[Chat templates](#chat-templates)).
Choose a context limit supported by the model. CPU testing can use
`--device cpu --dtype float32`.

`--device auto` offloads every layer to any llama.cpp backend compiled into the
wheel — Vulkan devices are used automatically when present, with CPU fallback.
`--device cpu` disables offload; `--n-gpu-layers N` sets an explicit layer count.
`--dtype` now selects the KV cache element type (`float16` default on GPUs,
`float32` for exact CPU scoring); GGUF weights keep their own quantization.

`--prefix-sharing` chooses how branches reuse the shared prompt. `auto` (the
default) shares prefix KV cells for ordinary attention models and switches
recurrent/hybrid architectures to per-branch prefill; `on` forces sharing for
every architecture; `off` forces per-branch prefill. See
[Recurrent and hybrid architectures](#recurrent-and-hybrid-architectures).

## API

See the [complete HTTP API reference](API_REFERENCE.md) for all request fields,
options, response formats, errors, limits, metrics and server arguments.

`POST /v1/classifier` and its alias `POST /v1/systemone` return non-streaming JSON.
`GET /health` reports readiness; `GET /v1/models` lists the served model and its configured Choice limit. `/docs` provides the generated API schema.
Use `--served-model-name` to set the public name (default: `--model`). Request model strings are not checked by default, so SDK defaults such as `jev-latest` work without selecting a different checkpoint. Add `--enforce-model-id` to require the served name. Responses always identify the served model.

```json
{
  "model": "/path/to/model.gguf",
  "messages": [{"role": "user", "content": "Mia owns a red bicycle. Her dog is named Max."}],
  "questions": {
    "color": {
      "type": "choice",
      "instructions": "What color is Mia's bicycle?",
      "criteria": {"red": null, "blue": null}
    },
    "dog": {"type": "noul", "instructions": "Is the dog named Max?"},
    "support": {
      "type": "score",
      "instructions": "How well does the context support that Mia owns a bicycle?",
      "criteria": ["Unsupported", "Supported"]
    }
  }
}
```

Supply exactly one of `messages` or `state`. State accepts text or JSON.
Chat uses the model's GGUF chat template (`tokenizer.chat_template` metadata,
or `--chat-template-file`). Choice supports up to 255 options by default;
`--max-choice-options` sets a cap from 2 to 255. Score still supports up to 50 levels.
Choice questions with at most 50 options retain their exact existing format. Larger
questions use exclusively two-letter uppercase labels, validated as distinct single
tokens, without mixing in single-letter labels. The GGUF loader checks that
capacity against the file's vocabulary (excluding control tokens) from a
vocabulary-only load, before any weights are loaded, and validates actual prompt
boundaries on each request. Unsupported vocabularies can use
`--max-choice-options 50`; options are never truncated. The server defaults to 100
scoring branches per request; `--max-request-branches` configures the limit. The
shared v1 template uses exactly one branch per question. Legacy choice/score
modes and score-format switches are no longer accepted.
Invalid input returns readable 422 errors; a full queue returns 429.

Responses contain `model`, `answers`, and `usage`. Answers include confidence.
`usage.input_tokens` counts unique token prefixes once, not the entire shared
context once per question. `usage.output_tokens` is zero: this implementation
reads logits without sampling any output tokens. Set
`ENABLE_OPEN_JEV_ADVANCED_METRICS=1` to include detailed timing and metadata.
Standard completion settings such as `max_tokens` and `temperature` are ignored.

## Configure request limits

Set these at server startup; they are independent:

| Flag | Default | Controls |
|---|---|---|
| `--max-request-branches` | `100` | Maximum questions per HTTP request (one branch per question). Set `256` for the schema maximum; larger values do not permit more than 256 questions. |
| `--max-model-len` | `16384` | Maximum tokens in each complete rendered branch: state/history, system and question instructions, options, template overhead, and repetitions. Not characters or output length. |
| `--max-choice-options` | `255` | Maximum options in each Choice question; valid settings 2–255. Score stays at 50 levels; Noul is unchanged. |

```bash
simple-jev --model /models/Qwen3.8-27B-Q4_K_M.gguf --device auto --dtype bfloat16 \
  --max-request-branches 256 --max-model-len 32768 --max-choice-options 255
```

This Qwen configuration automatically selects `shared_examples_binary` when no prompt
format is specified. Three 255-choice questions consume three branches, not 765.
The limits do not guarantee that all maxima fit simultaneously: long options and
policy repetition increase input length. Violations return 422; candidates/context
are never silently truncated. Setting a larger token cap does not extend the
checkpoint's native context support or guarantee sufficient memory:
`--max-model-len` sizes the llama.cpp KV pool and is clamped to the GGUF's
trained context length. The checked
255-choice prompts fit at 32K; other content may require more.

`--max-batch-size` and `--max-batch-tokens` control execution batches, not the
question-count limit. A single suffix must also fit the batch-token budget;
when using longer inputs, check that budget as well. Request `max_tokens` is not
an input-length setting and is ignored for this non-generating classifier.
See [the API limits reference](API_REFERENCE.md#limits-batching-and-cancellation).

## Prompt format selection

Omitting `--classifier-prompt-policy` auto-selects a development recommendation
for known language-backbone architecture/size profiles. Matching uses the GGUF
header (attention/expert dimensions and vocabulary), not file, repository, or
served names. Unknown profiles fall back to `baseline` with a prominent warning
to run `eval/prompt_search.py` first. Laya retains its native format.

GGUF files do not carry Hugging Face's `config.json`, so the server reads the
header's key/value metadata (`read_gguf_metadata`, including the per-layer arrays
that llama.cpp's own metadata view omits) and translates it into the same
fingerprint fields upstream reads from the HF text config
(`gguf_backbone_config`):

| Fingerprint field | GGUF key (`<arch>.` prefix) |
|---|---|
| `model_type` | `general.architecture`: `qwen35` → `qwen3_5_text`, `qwen35moe` → `qwen3_5_moe_text`, `gemma4` → `gemma4_text` or `gemma4_unified_text` (told apart by size) |
| `hidden_size`, `num_hidden_layers` | `embedding_length`, `block_count` minus `nextn_predict_layers` (MTP) |
| `num_attention_heads`, `num_key_value_heads` | `attention.head_count`, `attention.head_count_kv` (first sliding-window layer when per-layer) |
| `head_dim` | `attention.key_length_swa` when present (Gemma 4), else `attention.key_length` |
| `intermediate_size` | `feed_forward_length` |
| `num_experts`, `moe_intermediate_size`, `num_experts_per_tok` | `expert_count`, `expert_feed_forward_length`, `expert_used_count` |
| `vocab_size` | the GGUF vocabulary size |

The mapping follows llama.cpp's `convert_hf_to_gguf.py`; the header of llama.cpp's
Gemma 4 26B-A4B vocabulary GGUF resolves to the `Gemma MoE 26B-A4B` profile.
Quantization type is not part of the fingerprint. Any other architecture is
reported as `gguf:<arch>` in the warning and uses `baseline`.

Explicit selection always overrides auto-selection. In particular, explicit
`baseline` preserves the former default prompts, scoring, and text-chat support:

```bash
simple-jev --model /models/Qwen3.8-27B-Q4_K_M.gguf --device auto \
  --classifier-prompt-policy shared_examples_binary
```

| Reference model (any GGUF conversion of it) | Sharing-constrained policy | Native development correct /477 |
|---|---|---:|
| Qwen/Qwen3.8-27B | `shared_examples_binary` |445|
| Qwen/Qwen3.6-35B-A3B | `shared_repeat_state` |432|
| Qwen/Qwen3.5-4B | `shared_examples_binary` |374|
| google/gemma-4-26B-A4B-it | `shared_examples_binary` |437|
| google/gemma-4-12B-it | `shared_repeat_state` |427|

`shared_examples_binary` and `shared_repeat_state` keep system instructions and
native thinking flags uniform across mixed-type question branches. Type-specific
instructions follow shared state/chat. State repetition stays in the shared
prefix; chat turns are not duplicated. Both support text and [image chat](VISION.md) `messages`.
The experimental explicit-only `universal_shared` moves universal rules and the
complete labelled question catalogue before state/chat; only the selected ID and
answer prefix follow it, with no fixed thinking prefill. It remains an experimental
replay/tuning option, not an automatic recommendation. These development selections
reuse the quick477 cases; they are not fresh held-out benchmark scores.

Legacy formats remain explicit options for reproduction, but are excluded from
default tuning because mixed-type system instructions can defeat context sharing:

- `examples_binary`: strict decision rules, worked examples, raw text/pretty
  JSON state once, and binary no/yes Noul scoring.
- `repeat_state`: the same format with an explicitly marked second state copy.
- `strict_mix_repeat2`: strict rules, the entire user block twice, and the
  evaluated nine-bin Noul wording/scoring. No extra worked-example block.

The three legacy policies above accept **text/JSON `state` only**, not `messages`;
use `baseline` or a shared format to preserve chat turns. When the server is
started with the model's vision projector (`--mmproj`), chat may also carry
[image inputs](VISION.md); each image is encoded once per request and its
prefill is shared when the compiled context is shared. Tools remain unsupported.
Except for baseline/universal_shared, Choice branches prefill three fixed `[thinking]` lines through the model's native
chat template; Score/Noul branches answer directly. This does not generate
reasoning or output tokens. A template that drops the fixed prefill is rejected.
Binary Noul returns `{"type":"noul","noul":P(yes)}` in [0,1], with no nine-bin
0.01–0.99 remapping or nested rating diagnostics. Choice/Score math is unchanged.

These formats were selected in development-set prompt experiments, not held-out
evaluation. They are recommendations, not guaranteed optima for new revisions,
fine-tunes, quantizations, or precision settings. Unregistered sizes are not
guessed from nearby models. Advanced metadata records the resolved policy and
selection mode/profile (`prompt_policy`, `prompt_policy_selection`).

For a new model, use the [quick prompt search](../eval/README.md#prompt-format-search):

```bash
python eval/prompt_search.py --model /models/YOUR_MODEL.gguf --device auto \
  --max-model-len 32768 --output eval/results/my-prompt-search
```

Run this from the repository root in the server environment after preparing the
quick datasets. By default it evaluates baseline and the two sharing-compatible
named formats with native scoring. `--all-formats` opts into all seven formats,
including legacy and experimental candidates; `--policies` selects an explicit
subset. Apply the winner explicitly. Repeated-input policies need enough context (32K
fits the checked 255-option case), and larger requests use more memory.

Named policies pass text blocks to the model's native template while
leaving baseline rendering unchanged. The fixed assistant prefill is rendered with
`add_generation_prompt=False` and `continue_final_message=True`; llama-cpp-python's
Jinja formatter has no such option, so `LlamaCppTokenizer` reproduces Transformers
5.x exactly (a sentinel appended to the final message's last text block, then the
render is cut there, dropping the template's end-of-turn tokens). Prompt selection
does not establish numerical equivalence across execution environments or a
throughput guarantee.
Repetition consumes additional context; the complete rendered branch remains
subject to `--max-model-len`. Advanced metadata identifies
`hf-<policy>-v1` instead of the baseline `v1` template.

Named prompt formats change prompt/scoring adapters, not model precision or
execution controls: the llama.cpp context, KV cache reuse, batching, locking,
admission and inference code are shared by every format. Loading gains the header
fingerprint, a vocabulary-only probe, and the startup compile check below.
Image-specific loading and decoding are described in [VISION.md](VISION.md). No
worker, stream, kernel, or other performance optimizations are included. Laya
keeps its native formatting and rejects non-baseline prompt policies at startup.
Implementation: `hf_prompt_policies.py` (upstream's module, unmodified),
packaged alongside `hf_server.py`.

### Chat templates

Every format renders through the GGUF chat template. Before loading weights, the
server compiles a one-question-per-type sample request with the selected format,
so a template or vocabulary that cannot serve it stops startup with an explicit
message instead of failing every request. Typical causes:

- The named formats need a template that accepts OpenAI-style text-block
  `content` lists and renders assistant `reasoning_content` when
  `enable_thinking` is set. Qwen3.5's template does. Older templates (for
  example Qwen2.5's, or Gemma 4 templates that render reasoning only next to tool
  calls) do not, and are rejected just as upstream's Transformers server rejects
  a template that drops the fixed prefill.
- Answer labels must be single distinct tokens after the rendered answer prefix.

GGUF conversions often embed the template current at conversion time. Pass
`--chat-template-file newer.jinja` to use another one; it applies to every request,
and advanced metadata reports `chat_template_source`. Otherwise choose another
`--classifier-prompt-policy`; `baseline` works with any chat template.

## Shared-prefix execution

Image requests follow the same flow: image embeddings (encoded once per request
by the vision projector) occupy their expanded token positions in the prefix, a
shared prefix is never cut inside an image, and the text after each branch's last
image is packed like any other suffix; see [VISION.md](VISION.md).

The compiler calls `common.prepare_prompt(request, version="v1")`, assembles the
returned strings with state/chat roles, and applies the GGUF chat template.
See [the v1 specification](../common/PROMPT_STRUCTURE_V1.md). Compile each
question, find their exact common token prefix, and decode that prefix once on
sequence 0 of a unified llama.cpp KV pool. For each suffix batch, copy the
prefix cells into the batch's per-row sequences with `llama_memory_seq_cp`,
then pack all rows into one `llama_decode` call (no padding tokens) with logits
requested only at each row's final position. Row sequences are removed with
`llama_memory_seq_rm` after each batch, so cells and sequence ids stay reusable.
The prefix sequence remains untouched. Results return in request order.

Suffix batches are grouped longest-first, bounded by batch size, the packed
suffix token budget, and the remaining KV cells. Requests execute serially
against the context; parallelism is within each request. The pool is cleared
per request, with no persistent cross-request cache. The prefix prefill is one
decode and is not chunked by `--max-batch-tokens`.

### Recurrent and hybrid architectures

Architectures that carry rolling state — pure recurrent stacks such as Mamba or
RWKV, and hybrid attention/SSM stacks such as `qwen35` — report true from
`llama_model_is_recurrent` or `llama_model_is_hybrid`. They previously failed to
load. They now load and, under `--prefix-sharing auto`, run with **per-branch
prefill**: the common prefix is left empty, each branch decodes its own complete
prompt on its own sequence id, and rows are still packed into one `llama_decode`
with no padding. That strategy is valid for every architecture and costs only
the prefill it stops sharing.

Two limits change meaning when sharing is off, because each row's length becomes
its whole prompt rather than its suffix: `--max-batch-tokens` must be at least
the longest compiled prompt, and rows per decode are bounded by
`n_ctx // longest_prompt`. Size `--max-model-len` and `--max-batch-tokens` with
that in mind, or branches end up scored one per decode.

`--prefix-sharing on` keeps `seq_cp` sharing for these models. On
llama-cpp-python 0.3.35 with a hybrid `qwen35` GGUF, sharing agrees with
per-branch prefill about as closely as sharing already agrees with it on a dense
model, which indicates `llama_memory_hybrid::seq_cp` duplicates recurrent state
rather than aliasing it; [CHANGED.md §8](../CHANGED.md) records the measurements.
Pure recurrent architectures were not measured, so verify before relying on `on`
for one.

## Precision notes

The CPU backend is the numerical reference: on it, a branch decoded with the same
partition as an independent full-prompt decode matches it bit-for-bit (verified
by the backend and image tests). llama.cpp's CPU kernels are not batch-invariant,
though: packing rows together or splitting a prompt differently (shared prefix
plus a short suffix) can change rounding for longer prompts, by up to ~0.6 logits
measured on Qwen3-VL-2B-Q8_0 with ~500-token branches, for text and image
requests alike (see [VISION.md](VISION.md#numerical-behavior)).
GPU backends add kernel-level floating-point differences: on the Vulkan builds
validated here, per-logit deviations up to roughly 0.1 versus CPU were observed
even with a `float32` KV cache, and `float16` KV adds its own rounding. Answers
rarely change, but exact parity with CPU or with the previous Transformers
implementation is **not** guaranteed on GPU. Use `--device cpu --dtype
float32` when scoring must be numerically anchored; treat GPU results as
approximately equal, not identical.

With a GPU-enabled wheel (for example Vulkan), `--device cpu` keeps the weights
on the host but llama.cpp still *offloads operations* for batches of 32 or more
tokens to the GPU (`op_offload`, on by default): the startup log shows a
non-zero `Vulkan0 compute buffer`. The server keeps llama.cpp's default, so
results stay comparable with earlier runs on the same machine; for a strictly
CPU reference use a CPU-only `llama-cpp-python` build. The real-engine fidelity
test disables op offload itself when `SIMPLE_JEV_DEVICE=cpu`.

Recurrent and hybrid models (e.g. `qwen35`) have one more batch effect, on CPU
too: llama.cpp slices packed sequences into equal-length sub-batches, so a
question row packed with longer rows has its recurrent state computed in several
pieces instead of one pass. Its scores then differ slightly from scoring that
prompt alone (about 0.03 logits on Qwen3.5-0.8B-BF16), with either
`--prefix-sharing` mode. Scores are still deterministic for the same request
and settings; rows of equal length are unaffected. On Windows consoles, set
`PYTHONUTF8=1` to avoid harmless `UnicodeEncodeError` warnings from
llama-cpp-python's log callback.

## Scope and validation

This reference accepts text and **image chat**. Image input needs the GGUF
vision projector that matches the text model (`--mmproj`, e.g. `mmproj-F16.gguf`
from the same repository) and runs through llama.cpp's libmtmd, so any projector
family llama.cpp supports can be used: Qwen2/2.5/3-VL and Qwen3.5 (M-RoPE),
Gemma 3 (non-causal image attention), SmolVLM/Idefics3 (tiled images), LLaVA-style
projectors and more. Images may be inline PNG/JPEG/WebP data URLs or bounded public
HTTP(S) URLs. Audio, video and tools are rejected. See [transport security,
limits, decoding and numerical behavior](VISION.md). Without `--mmproj`, image
requests return HTTP 422.

Optional `--max-image-width` / `--max-image-height` set hard **resize** caps before
the projector's own preprocessing. `--default-image-max-width` /
`--default-image-max-height` set request defaults; `media_io_kwargs.image.max_width/max_height`
overrides them within the hard caps. Oversized images are downscaled to fit,
preserving aspect ratio—not rejected for exceeding these dimensions. Original
safety limits still apply, and the projector may subsequently resize/pad to its
required grid.

Models must be GGUF files with a `tokenizer.chat_template` and single-token
rating/choice labels at the assistant boundary. Recurrent and hybrid
architectures are accepted and default to per-branch prefill, described in
[Recurrent and hybrid architectures](#recurrent-and-hybrid-architectures).
Arbitrary model compatibility is not guaranteed.

The shared v1 prompt and scoring rules are the source of truth for the explicit
`baseline` format; alternative policy differences are described above.
It does not claim exact numeric equivalence with another inference engine.
Tests verify the orchestration against a stubbed engine — prefix reuse, per-row
sequence copies, packed batches, cleanup, metrics, cancellation — plus the
loader (GGUF header fingerprinting, vocabulary-only probe, served names, Choice
capacity, prompt formats, template override) against a stubbed llama.cpp loader,
and `continue_final_message` rendering (compared with Transformers when it is
installed). When `SIMPLE_JEV_GGUF` points at a local GGUF file, they also compare
shared-prefix scores against independent full-prompt decodes on the real engine
and run end-to-end HTTP checks, including discovery and a 64-option Choice.
`SIMPLE_JEV_DEVICE` (default `cpu`) pins the device for those runs. They
establish execution equivalence, not answer quality.
Image tests (`tests/test_llama_vision.py`) cover marker rendering, per-image
spans, label boundaries, M-RoPE positions, once-per-request encoding, both
prefill strategies, loading and HTTP against a stubbed projector and engine; with
`SIMPLE_JEV_MMPROJ` also set, they compare the compiler's token layout with
`mtmd_tokenize` and the backend's scores with llama.cpp's own
`mtmd_helper_eval_chunks` evaluation (see [VISION.md](VISION.md#validation-scope)).
Image-validation, resize and public-URL download tests come from upstream unchanged.

```bash
python -m pytest -c hf-server/pyproject.toml common/tests hf-server/tests -q
```

## Laya backend

Install the optional SDK and select the backend explicitly:

```bash
pip install -e './hf-server[laya,test]'
USE_TF=0 python hf-server/hf_server.py \
  --backend laya --model convaiinnovations/laya --device cpu
# Add --subfolder multilingual or --subfolder typed-decisions for those checkpoints.
```

`--backend llama-cpp` remains the default; GGUF model commands are unchanged.
Laya loads once per process and uses its own
encoder/option-marker format, not the common v1 assistant-prefill template.
`--revision` selects the downloaded checkpoint revision. `--device auto` lets the
SDK select CUDA, MPS, or CPU; `--dtype` and the llama.cpp suffix batching
controls apply only to the GGUF backend. Laya uses its SDK precision policy and
batches the admitted questions together; `--max-request-branches` bounds that
batch.

The endpoint and `ClassifierRequest` stay the same. Send the repository ID as
`model`, including when a subfolder is selected at startup. Text `messages` are
serialized as a list of role/content objects. Images, tools, message extras, and
raw-logit diagnostics are rejected. Context overflow is rejected before inference;
the effective limit is the smaller of `--max-model-len` and the checkpoint's native
`max_len`. Laya's native formatter still budgets/truncates question and option text
according to its own `head_max_len` rules.

Choice/score probabilities retain the SDK's temperature scaling, confidence is
the largest returned probability, and Noul is its native binary positive-class
probability (not the v1 nine-bin mapping). SDK action fields are omitted. Token
usage sums the actual per-question sequences, so repeated context is counted;
output tokens are zero. Advanced metadata identifies the format as `laya-native`.
The shared admission queue and a model lock bound concurrent work; cancellation
cannot interrupt an already running engine forward.

Upstream validation of the Transformers server on macOS (2026-09-20), retained
for the Laya results: 50 common/server tests passed, including tiny
Qwen/Gemma backend regression tests and Laya adapter validation. Real weights for
`convaiinnovations/laya` and `Qwen/Qwen3.5-0.8B` both returned HTTP 200 through the
ASGI classifier route on CPU for a combined choice/score/Noul request, and rejected
an incorrect model ID with HTTP 422. Laya used 93 input tokens; Qwen used 751.
This validates integration, not accuracy: the small Qwen answered the example's
Noul question incorrectly. Direct MPS loading of Qwen crashed natively on this
Mac, so this run does not establish MPS compatibility. Laya's multilingual subfolder loading is covered by adapter tests, not real-weight
inference in this validation run. Typed Decisions extension validation is below.


### Experimental 2× Laya RoPE interpolation

For the ModernBERT Typed Decisions checkpoint:

```bash
USE_TF=0 python hf-server/hf_server.py \
  --backend laya --model convaiinnovations/laya \
  --subfolder typed-decisions --device cpu \
  --rope-factor 2 --max-model-len 2048
```

This halves full-attention and sliding-attention rotary inverse frequencies,
so position p has the original rotary angle at p/2. It doubles the checkpoint's
native sequence budget (1,024 → 2,048 here); the admission limit still respects
`--max-model-len`. It does not enlarge the sliding attention window. The flag
only supports unscaled ModernBERT RoPE and rejects other backends/architectures.
The default factor is 1, preserving existing model behavior. This is experimental:
long-input execution is not evidence of accuracy or calibration, and interpolation
also changes behavior on shorter inputs. No weights or downloaded configs are
modified; changes apply to the loaded process only.

Extension validation: all 52 tests passed. Real Typed Decisions weights with 2×
interpolation returned HTTP 200 for a 1,417-token sequence and chose the requested
refund category. An oversized sequence returned HTTP 422 at the 2,048-token limit.
This is an execution smoke test, not a long-context accuracy benchmark.


### General RoPE extension

`--rope-factor 2` enables experimental linear position interpolation for either
backend. The default is 1 (no change). Finite factors greater than 1, including
fractional factors (for example `--rope-factor 1.5`), are accepted. Rotary scaling
uses the exact factor; resulting token capacities are rounded down to integers.
`--laya-rope-factor` remains a CLI alias.

```bash
python hf-server/hf_server.py \
  --model Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  --gguf-file qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --device cpu --dtype float32 \
  --rope-factor 2 --max-model-len 2048
```

For the GGUF backend, the server configures llama.cpp's native linear rope
scaling on the context (`rope_scaling_type=LINEAR`, `rope_freq_scale=1/factor`),
which is the same scaling the `llama-server --rope-scale` flag applies. Laya
retains the ModernBERT implementation described above and doubles its checkpoint
sequence budget at 2×. `--max-model-len` remains the independent input admission
limit: RoPE scaling does not multiply this setting. These options are
experimental, not a promise that
all Hugging Face architectures support extended context or retain model quality.
