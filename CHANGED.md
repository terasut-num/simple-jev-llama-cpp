# CHANGED — GGUF / llama.cpp Refactor of the Classifier Server

Date: 2026-09-22
Scope: `hf-server/` inference engine replacement — Hugging Face Transformers + PyTorch → llama.cpp (GGUF) via `llama-cpp-python`, with Vulkan GPU acceleration.
Reference upstream: <https://github.com/featherless-ai/simple-jev/tree/main>

---

## 1. What was replaced

| Aspect | Before | After |
| --- | --- | --- |
| Inference engine | Transformers `AutoModelForCausalLM` / `AutoModelForImageTextToText` + PyTorch | llama.cpp via `llama-cpp-python` (`_internals.LlamaModel` / `LlamaContext`) |
| Model format | HF safetensors directories / repo ids | GGUF files, GGUF-containing directories, or HF GGUF repositories |
| Acceleration | `device_map` (CUDA/ROCm/CPU) | `n_gpu_layers` offload; Vulkan devices used automatically, CPU fallback |
| Shared-prefix reuse | `past_key_values` deep-copies + `reorder_cache` | Unified KV pool + `llama_memory_seq_cp` per row sequence |
| Suffix batching | Padded `[rows, width]` tensors + attention masks | Padding-free packed `llama_batch` rows |
| Tokenizer | `AutoTokenizer` | llama.cpp native GGUF vocabulary |
| Chat template | `tokenizer.apply_chat_template` | GGUF `tokenizer.chat_template` metadata rendered with llama-cpp-python's `Jinja2ChatFormatter` |
| KV precision | `torch.dtype` for weights and cache | `--dtype` selects KV cache element type only (weights keep GGUF quantization) |
| RoPE extension | Transformers `rope_parameters` config rewrite | llama.cpp context `rope_scaling_type=LINEAR`, `rope_freq_scale=1/factor` |

**Preserved without change** (core classification logic and API contracts):
- `common/` request validation, v1 prompt building, label semantics, softmax/scoring math, response envelope
- HTTP surface: `POST /v1/classifier`, alias `POST /v1/systemone`, `GET /health`, `/docs`; 422/429/499 error shapes and headers
- Metrics key names (`prefix_tokens`, `suffix_batch_sizes`, `engine_forwards`, `branch_prompt_tokens`, `computed_prompt_tokens`, `logical_prefill_tokens`, `padded_suffix_tokens`, `branch_output_tokens=0`, `scored_positions`, `backend_seconds`, `queue_seconds`, `total_seconds`, `prefill_strategy="shared_prefix"` — which later gained a second value, see §8)
- Usage accounting: unique token-prefix union for `input_tokens`, `output_tokens` always 0
- Advanced-metrics opt-in (`ENABLE_OPEN_JEV_ADVANCED_METRICS`, `raw_logits`)
- Queue/concurrency model: concurrency 1, 16 queued slots, cooperative cancellation between batches under the model lock
- `PromptCompiler`'s duck-typed tokenizer interface (`encode(text, add_special_tokens=False)`, `apply_chat_template(..., tokenize=False)`) — external consumers such as `RFDT/train.py` and `common/tests/test_pipeline.py` are unaffected
- The optional Laya backend (`--backend laya`) including its RoPE extension

---

## 2. New backend design (`LlamaCppBackend`)

The shared-prefix execution strategy was translated to llama.cpp's multi-sequence API:

1. **Per request** the KV pool is cleared (`llama_memory_clear`), so no state leaks between callers.
2. The branches' exact common token prefix is decoded **once, unchunked** on sequence 0 (`llama_batch` with no logits requested).
3. For each suffix batch: `llama_memory_seq_cp(memory, 0, row+1, 0, len(prefix))` marks the shared prefix cells for each row's own sequence id — attention then sees prefix + suffix without recomputing prefill.
4. All rows of the batch are packed into **one** `llama_decode` call: no padding tokens, explicit `pos` continuing after the prefix, `logits[i]` flagged only at each row's final suffix token.
5. `llama_get_logits_ith` is read at each row's flagged batch position; only the permitted label token ids are converted to plain Python floats, keyed by the plan's exact output labels.
6. `llama_memory_seq_rm(memory, row+1, 0, -1)` after each batch drops the rows' claims on shared cells and frees their suffix cells, making both cells and sequence ids reusable for the next batch.

Steps 2 and 3 are skipped when prefix sharing is disabled (§8): the prefix is empty, each branch decodes its complete prompt on its own sequence id, and the rest of the loop is unchanged.

Batching keeps the previous engine's shape: suffixes sorted longest-first, rows bounded by `--max-batch-size`, by `max_batch_tokens // width` (width of the longest suffix in the batch), and by the remaining KV cell budget `(n_ctx - prefix) // width`.

Backend results changed from 1-D torch tensors to **label→float mappings** per branch, one of the forms `common.response_scoring` already accepts, removing the torch dependency from the scoring path.

Context parameters chosen by the loader:

| Parameter | Value | Why |
| --- | --- | --- |
| `n_ctx` | `--max-model-len` (clamped to `n_ctx_train`) | admission limit and KV pool size |
| `n_batch` | `max(max_batch_tokens, max_model_len)` | one decode must hold the prefix or a full token budget; llama.cpp additionally clamps it to `n_ctx` |
| `n_seq_max` | `max_batch_size + 1` | sequence 0 (prefix) plus one row sequence per batch slot |
| `kv_unified` | `True` | partial-range `seq_cp` requires the unified cache; the split cache asserts |
| `type_k` / `type_v` | `--dtype` mapped | KV cache element type |
| `rope_*` | from `--rope-factor` | linear position interpolation |

---

## 3. Empirical findings (probe work before implementation)

The engine swap was preceded by probe scripts against a real GGUF to pin down llama.cpp semantics. These findings drove the design:

1. **CPU is the numerical reference.** On CPU, shared-prefix scoring (`seq_cp` + packed suffix decode) equals independent full-prompt decodes **bit-for-bit** (max diff 0.0). Chunked decode equals single-shot decode exactly on CPU as well.
2. **Vulkan kernel divergence.** This build's Vulkan backend deviates from CPU by up to ~0.1 logits per token for multi-chunk decodes (prefill-then-continue), regardless of flash-attention on/off and KV dtype (f16 or f32). Single-decode-per-batch scoring is therefore used throughout; remaining GPU-vs-CPU differences are documented as a precision caveat. Live CPU and Vulkan runs produced agreeing answers (see §6).
3. **`llama_get_logits_ith(ctx, i)` indexes batch token positions**, not output ordinals. `i` must be a position flagged in the batch (`batch.logits[i] == true`), otherwise the call returns NULL and ctypes raises.
4. **Partial-range `llama_memory_seq_cp` requires `kv_unified=True`.** The default split cache aborts with `GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers")`.
5. **Unified KV pool = `n_ctx` cells total**, shared across sequences by per-cell sequence bitmasks. `n_batch` is silently clamped to `n_ctx`. Requested `n_ctx` is rounded up by llama.cpp (e.g. 64 → 256; 1000 → 1024); the backend reads `context.n_ctx()` for the real budget.
6. **Per-sequence positions must remain consecutive across decodes** (`Y = X + 1`). Re-decoding a sequence from position 0 without clearing fails; independent reference decodes need fresh sequence ids or a clear.
7. **`llama_batch` memory is recycled and uninitialized.** Every field of every used slot must be set; failures surface as misleading "invalid token[i] = <garbage>" messages. Batch arrays are filled through cached ctypes handles to avoid repeated pointer fetches.
8. **`n_outputs_max_per_seq` defaults to 1** — one flagged output per sequence per decode is the safe contract for multi-row scoring batches.
9. **ggml_type values**: F32=0, F16=1, BF16=30. The constants are not re-exported by `llama_cpp.py`, so `KV_CACHE_TYPES = {"float32": 0, "float16": 1, "bfloat16": 30}` maps `--dtype` to `type_k`/`type_v`.
10. **Recurrent/hybrid architectures were taken to be unable to share prefixes** — `llama_memory_seq_cp` was assumed unable to copy their rolling state — so the loader rejected them outright via `llama_model_is_recurrent` / `llama_model_is_hybrid`. This finding was re-measured against a real hybrid GGUF and did not hold; **superseded by §8**.
11. **GGUF chat templates** live in `tokenizer.chat_template` metadata and render correctly with llama-cpp-python's `Jinja2ChatFormatter` (trim_blocks/lstrip_blocks and the `{% generation %}` pass-through match HF). The formatter bakes `add_generation_prompt` in at construction, so one cached formatter is built per flag value; `enable_thinking` and other HF-style kwargs are forwarded into the render, mirroring Transformers.
12. **Tokenization parity**: `LlamaModel.tokenize(text_bytes, add_bos, special)` with `add_bos=False` mirrors HF `encode(add_special_tokens=False)` when the rendered template already carries its special tokens.

---

## 4. Files changed

### `hf-server/hf_server.py` — the core refactor

- **Removed**: `HFBackend`, all torch/Transformers imports, `configure_rope`'s Transformers implementation, the Transformers branch of `load_service`, `copy`/`inspect` imports.
- **Added**:
  - `LlamaCppBackend` — engine described in §2, with the same async/cancellation/locking contract as before (`asyncio.to_thread` + cooperative stop event + thread lock held through worker exit).
  - `LlamaCppTokenizer` — GGUF-native `encode` / `apply_chat_template` adapter for `PromptCompiler`.
  - `resolve_gguf_path` — local `.gguf` file, directory with exactly one `.gguf` (shard sets count as one), or HF repository download via `huggingface_hub`; `--gguf-file` selects one file from a multi-variant repository and avoids downloading every quantization. The repository id remains the request-visible model name.
  - `resolve_n_gpu_layers` — maps `--device` (`cpu` → 0 layers; `auto`/`gpu`/`vulkan`/… → all layers) with `--n-gpu-layers` overriding.
  - `KV_CACHE_TYPES` — dtype → ggml_type mapping for the KV cache.
  - `configure_rope` (rewritten) — llama.cpp linear rope scaling on the context params.
- **`load_service`**: `backend` default is now `"llama-cpp"`; the GGUF branch loads model + context with the parameters from §2, rejects unsupported dtypes/devices, and reports metadata: `backend`, `model_revision`, `gguf_path`, `n_gpu_layers`, `kv_cache_dtype`, `rope_factor`. It originally also rejected recurrent/hybrid models; §8 replaced that with a prefill-strategy choice and added the `stateful_architecture` and `prefix_sharing` metadata fields. The Laya branch is unchanged.
- **`main()`**: `--backend` choices are `["llama-cpp", "laya"]`; new `--n-gpu-layers`; `--dtype` help documents its KV-cache meaning. `--device`/`--dtype`/`--max-batch-tokens` keep their names for command compatibility. `--prefix-sharing` was added later by §8.
- Metrics: `backend` is `"llama-cpp"`; because rows are packed without padding, `computed_prompt_tokens == logical_prefill_tokens` and `padded_suffix_tokens` equals the packed suffix total (key names retained for API stability, meanings documented).

### `hf-server/pyproject.toml`

- Dependencies: `torch`, `transformers`, `accelerate` → `llama-cpp-python>=0.3.35`; `jinja2` added explicitly (used by the chat formatter). Description updated. Laya and test extras unchanged.

### Tests

- **Removed** `hf-server/tests/test_hf_backend.py` (torch/tiny-Transformers-model regression suite).
- **Added** `hf-server/tests/test_llama_backend.py`:
  - Stubbed-engine orchestration tests (no weights): prefix decoded once on seq 0, exact per-row `seq_cp` ranges, packed rows with one flagged position each, `seq_rm` cleanup and sequence-id reuse, width-based batch splitting, identical-prompt suffix reservation, decode-failure and plan-required errors, cancellation contract, metrics values.
  - Opt-in real-engine tests gated by `SIMPLE_JEV_GGUF` (local GGUF path) and `SIMPLE_JEV_DEVICE` (default `cpu`): backend shared-prefix scores vs independent full-prompt decodes on the same context, plus an end-to-end HTTP check through `create_app` covering choice/score/noul, usage, and answer shape.
- **Updated** `hf-server/tests/test_laya.py`: removed the Transformers-specific `configure_rope` tests; added `test_llama_cpp_loader_rejects_bad_inputs`, `test_llama_cpp_rope_configuration`, `test_resolve_gguf_paths`; added a skip guard for the torch-dependent Laya RoPE numerical test (torch remains optional for the Laya path only).

### Documentation

- `hf-server/README.md`: GGUF/Vulkan install instructions (`CMAKE_ARGS="-DGGML_VULKAN=ON"`), model resolution rules, rewritten shared-prefix execution section, new **Precision notes** section, scope and validation updates, Laya/RoPE sections updated.
- `hf-server/API_REFERENCE.md`: backend/metadata/metrics tables (`llama-cpp`, new metadata fields, packing-aware metric meanings), batching/limits section, GPU precision note, full CLI argument table including `--n-gpu-layers` and the new `--device`/`--dtype` semantics.
- Root `README.md`: intro, "Running the GGUF Server" instructions, GPU example (Vulkan offload instead of `--device cuda`), explicit `--gguf-file` selection for multi-variant Hugging Face repositories, example model identifiers, component table, prompt-flow description, and testing/model-requirements sections.

---

## 5. CLI changes

| Flag | Change |
| --- | --- |
| `--backend` | choices `llama-cpp` (new default), `laya`; `transformers` removed |
| `--model` | now a GGUF path / GGUF directory / HF GGUF repository id |
| `--device` | `cpu` = no offload; `auto`/`gpu`/`vulkan` = offload all layers via any compiled llama.cpp backend (Vulkan first when present); unknown values rejected |
| `--dtype` | selects the KV cache element type (float32/float16/bfloat16); weights keep their GGUF quantization |
| `--n-gpu-layers` | new; explicit llama.cpp offload count, overrides `--device` |
| `--revision` | used for the HF GGUF download |
| `--gguf-file` | new; selects one `.gguf` from a Hugging Face repository with multiple variants, downloading only that file |
| `--prefix-sharing` | added by §8; `auto` (default) / `on` / `off` selects the prefill strategy, replacing the hard rejection of recurrent/hybrid architectures |
| others (`--max-model-len`, `--max-batch-size`, `--max-batch-tokens`, `--max-request-branches`, `--rope-factor`, `--subfolder`, `--host`, `--port`) | names unchanged; `--max-model-len` additionally sizes the KV pool and is clamped to the trained context |

---

## 6. Validation results

- **Test suite**: `python -m pytest -c hf-server/pyproject.toml common/tests hf-server/tests -q` → **62 passed, 2 skipped** by default (the 2 real-engine tests skip without `SIMPLE_JEV_GGUF`).
- **Real-engine tests** (with `SIMPLE_JEV_GGUF` pointing at a local Qwen2.5-0.5B-Instruct Q2_K GGUF): **64 passed** on CPU, including the shared-prefix-vs-independent fidelity test (CPU worst diff < 1e-3, measured 0.0) and the end-to-end HTTP test; the HTTP test also passed with `SIMPLE_JEV_DEVICE=vulkan`.
- **Live server, CPU** (`--device cpu --dtype float32 --max-model-len 2048`): combined choice/score/noul request returned HTTP 200 with `choice: "red"` (confidence 0.950), score 0.919 ("Fully supported"), `usage.input_tokens: 715`, `output_tokens: 0`; wrong model id → 422; `/health` → 200.
- **Live server, Vulkan** (`--device auto --dtype float16`): all 24 layers offloaded to the NVIDIA RTX GPU (`Vulkan1` KV buffer, flash attention enabled); the same request returned `choice: "red"` (0.948), score 0.9194 — agreeing answers with small GPU-vs-CPU logit differences consistent with the documented precision caveat.
- `--help` still loads no weights (heavy imports remain local to `load_service`).

### Test-environment notes

- The project `.venv` (Python 3.14) was extended with a prebuilt Vulkan-enabled `llama-cpp-python` 0.3.35 wheel (contains `ggml-vulkan.dll`) plus its `diskcache` dependency. Building from source requires CMake + MSVC and `CMAKE_ARGS="-DGGML_VULKAN=ON"`; no C toolchain is present on this machine, so the prebuilt wheel was reused.
- A temporary `.probe/` directory used for engine-mechanics validation was removed after the refactor; nothing in the package depends on it.

---

## 7. Known caveats

1. **GPU precision**: exact parity with CPU (and with the previous Transformers implementation) is not guaranteed on GPU backends; this build's Vulkan kernels showed up to ~0.1 per-logit deviation for chunked decodes even with an f32 KV cache. Use `--device cpu --dtype float32` when scoring must be numerically anchored. Answers agreed in all live validation runs.
2. **Model requirements**: GGUF with `tokenizer.chat_template` and single-token-stable answer labels at the assistant boundary (checked per branch by `PromptCompiler` as before). GGUFs lacking a chat template are rejected at load. Recurrent/hybrid architectures are supported as of §8, with per-branch prefill by default.
3. **Laya**: unchanged, still requires its optional SDK and (for the RoPE numerical test) torch; both install extras remain.
4. **`--backend transformers` is gone**: scripts passing it will now get an `Unknown backend` error.

---

## 8. Follow-up: prefix sharing on recurrent/hybrid architectures

Date: 2026-09-22
Scope: `hf-server/hf_server.py`, `hf-server/README.md`, root `README.md`.

### 8.1 Why

Loading a hybrid GGUF (`unsloth/Qwen3.5-4B-GGUF`) failed at startup:

```
ValueError: This GGUF architecture has recurrent/hybrid state and cannot share
a prefix KV cache
```

That guard implemented §3 finding 10. The GGUF metadata confirms the
classification — `general.architecture = qwen35`, carrying `qwen35.ssm.*` keys
and `qwen35.full_attention_interval = 4`, i.e. SSM layers interleaved with a
full-attention layer every fourth block — so `llama_model_is_hybrid` is right
about the model. The open question was whether the *conclusion* still held on
the current engine.

### 8.2 Measurement

Probed with llama-cpp-python 0.3.35 (Vulkan-enabled wheel, `n_gpu_layers=0`),
comparing each strategy's **full logit vector** against a single-sequence,
full-prefill baseline. Repeating that baseline reproduces bit-for-bit
(`max|Δlogit| = 0.0000`), so any deviation below is real rather than jitter.

Four branches over a shared 158-token prefix, 18–22 token suffixes:

| Strategy | Dense Qwen2.5-0.5B *(already supported)* | Hybrid Qwen3.5-4B |
| --- | --- | --- |
| One sequence per decode | 0.0000 | 0.0000 |
| 4 sequences packed, **no sharing** | 0.2807 | 0.3589 *(1 of 4 argmax flipped)* |
| 4 sequences packed, `seq_cp` **shared prefix** | 0.3042 | 0.4117 |

The middle row is the decisive one: the strategy that shares nothing drifts just
as far. The spread tracks **multi-sequence batch shape**, not prefix sharing,
and is already present on a dense model this server ships. Sharing adds little
on top. Two controls agreed: reversing branch order within the batch changed
nothing material (0.4019), and reaching the same state via
`llama_state_seq_get_data` / `set_data` instead of `seq_cp` landed in the same
band (0.3774).

A second probe replicated `LlamaCppBackend._score` exactly — prefill seq 0 once,
then per batch `seq_cp` → decode → `seq_rm` — with 8 branches over a 261-token
prefix and ~50-token suffixes, split into two batches of four. This is where
aliased state would surface, since batch 2 re-copies from a seq 0 that batch 1
may have consumed:

```
batch 1 worst |Δlogit| = 0.4466
batch 2 worst |Δlogit| = 0.1927      argmax matched baseline on all 8 branches
```

Batch 2 is *better* than batch 1. The prefix state survives re-copying, so
`llama_memory_hybrid::seq_cp` duplicates recurrent state rather than aliasing
it, and §3 finding 10 no longer describes this engine.

**Not measured**: a pure recurrent model (`llama_model_is_recurrent` true, e.g.
Mamba or RWKV). Only the hybrid case was exercised, which is why the default
stays conservative.

### 8.3 What changed

- `LlamaCppBackend.__init__` takes `share_prefix=True`. When false, `_score`
  leaves the common prefix empty, skips the `seq_cp` loop, and lets every branch
  prefill its complete prompt on its own sequence id. Rows are still packed into
  one padding-free `llama_decode`, and cleanup is unchanged.
- `prefill_strategy` in the metrics reports `per_branch` in that mode, alongside
  the existing `shared_prefix`.
- `load_service` no longer raises for recurrent/hybrid models. It computes
  `stateful` from `llama_model_is_recurrent` / `llama_model_is_hybrid` and
  resolves a new `prefix_sharing` argument: `auto` (default) disables sharing
  for stateful architectures and keeps it everywhere else, `on` forces sharing,
  `off` forces per-branch prefill. An unknown mode raises.
- Loader metadata gains `stateful_architecture` and `prefix_sharing`.
- `main()` exposes `--prefix-sharing {auto,on,off}`.

Behaviour for ordinary attention models is unchanged: `auto` resolves to
`share_prefix=True`, which is the original code path.

### 8.4 Operational note

With sharing off, each row's length is its whole prompt rather than its suffix,
so two existing limits change meaning: `--max-batch-tokens` must be at least the
longest compiled prompt (it is validated against row length), and rows per
decode are bounded by `n_ctx // longest_prompt`. Running
`--max-model-len 4096 --max-batch-tokens 4096` therefore scores one branch per
decode. Raise `--max-model-len` above the real prompt length, or use
`--prefix-sharing on`, to keep branches batched.

### 8.5 Validation

- `python -m pytest tests -q` from `hf-server/` → **36 passed, 11 skipped**,
  identical to the pre-change result; no test covered the removed guard.
- The hybrid GGUF loads successfully under all three modes, with metadata
  reporting `stateful_architecture=True` and `prefix_sharing` following the
  selected mode.
- `--help` still loads no weights.

---

## 9. Upstream sync: model discovery, prompt formats, 255-option Choice, quick-eval search

Date: 2026-09-24
Scope: merge of `featherless-ai/simple-jev` `main` up to `5686b21` ("Add model-aware prompt defaults, quick search, and classifier limits"), including the earlier upstream commits this fork had not yet taken (evaluation suites and audits, named prompt policies, website/evaluation pages). The llama.cpp/GGUF engine from §1–§8 is unchanged; upstream's Transformers-specific loading was translated to GGUF.

### 9.1 Taken from upstream unchanged

- `common/`: Choice accepts 2–255 options; above 50, `prepare_prompt` takes adapter-supplied, tokenizer-validated two-letter labels (`AA`, `AB`, …). Requests with ≤50 options render exactly as before.
- `hf-server/hf_prompt_policies.py` (new, unmodified): the `baseline`, `examples_binary`, `repeat_state`, and `strict_mix_repeat2` formats, binary Noul restoration, and the architecture/size profile table (`KNOWN_PROFILES`) with `resolve_prompt_policy`.
- Service/HTTP behaviour in `hf_server.py`: `GET /v1/models` discovery (no inference slot, `x_max_choice_options`), a public `--served-model-name`, permissive request model IDs by default with opt-in `--enforce-model-id` (error text is now `Served model is …`), responses always naming the served model, `--max-choice-options` (2–255) enforced independently of `--max-request-branches`, `PromptCompiler` prompt policies/extended labels, and `Branch.reasoning_content` / `CompiledRequest.binary_noul_keys`.
- `eval/`: full suites, presets, audits, comparisons, the sequential quick-eval `prompt_search.py`, and their offline tests; `.github/workflows/test-evaluations.yml`; `scripts/export_eval_page.py`; website evaluation pages.
- Upstream tests `test_auto_policy.py`, `test_discovery_choices.py`, `test_prompt_policies.py`, and the `test_api.py`/`test_laya.py` updates.

### 9.2 Adapted for GGUF / llama.cpp

| Upstream (Transformers) | This fork (GGUF) |
| --- | --- |
| Profile read from HF `config.json` via `AutoConfig` | `read_gguf_metadata` parses the GGUF header (keeping the per-layer arrays llama.cpp's own metadata view drops) and `gguf_backbone_config` maps it onto the same fingerprint fields: `qwen35`/`qwen35moe`/`gemma4` → HF text model types, `block_count − nextn_predict_layers`, `key_length_swa` as Gemma 4's `head_dim`, sliding-layer `head_count_kv`, `expert_*`, vocabulary size. Unprofiled architectures appear as `gguf:<arch>` and fall back to `baseline` with upstream's warning. |
| `tokenizer.apply_chat_template(..., continue_final_message=True)` | `LlamaCppTokenizer` reproduces Transformers 5.x exactly: a sentinel appended to the final message's last text block, render, cut at the sentinel. `enable_thinking` and `reasoning_content` pass through to the GGUF Jinja template. |
| `tokenizer.all_special_ids` | `LlamaCppTokenizer.all_special_ids`: tokens with the llama.cpp `CONTROL` attribute, scanned once on first use. |
| Choice capacity checked before `from_pretrained` loads weights | A vocabulary-only llama.cpp load (`vocab_only=True`) fingerprints the header, resolves the policy, and checks two-letter label capacity before any weights load. |
| A template that drops the fixed prefill is rejected per request | The same rejection happens at startup: a sample request (`STARTUP_PROBE_REQUEST`) is compiled with the resolved policy, and failure stops startup with guidance. |
| `prompt_search.py` launches the Transformers server | It launches the GGUF server; adds `--gguf-file`, `--n-gpu-layers`, `--prefix-sharing`, `--chat-template-file` (all fixed across formats); local `.gguf` paths stay local and are recorded by size; offline pinning looks up the cached `--gguf-file`; provenance lists `llama-cpp-python` instead of torch/transformers. |

New in this fork: `--chat-template-file` replaces the GGUF's embedded template for every request (metadata `chat_template_source`), because GGUF conversions often embed templates that predate the reasoning prefill. `load_service` now validates `--prefix-sharing` and `--device` before touching any file, and closes the model if context creation fails. Advanced metadata adds `prompt_policy`, `prompt_policy_selection`, and `chat_template_source`.

Behaviour change to note: with no `--classifier-prompt-policy`, a recognized Qwen3.5/Gemma 4 GGUF now starts with its recommended named format, which accepts `state` only. Pass `--classifier-prompt-policy baseline` to keep the previous prompts and chat `messages` support. Unrecognized models (for example Qwen2.5) behave as before, after a startup warning.

### 9.3 Tests

- The three upstream loader tests that mocked `transformers.Auto*.from_pretrained` were rewritten against `conftest.fake_gguf_loader`, which records llama.cpp model/context parameters while `load_service` runs its real control flow.
- New `test_gguf_policies.py`: header reader (arrays kept, string arrays skipped, bad magic), all five profiles resolved from converter-style GGUF headers (plus near-size and unknown-architecture fallbacks), `continue_final_message` trailing-space and error handling, parity with Transformers' renderer when `transformers` is installed, control-token IDs, startup probe rejection without loading weights, template override, and CLI pass-through.
- New real-engine test (opt-in `SIMPLE_JEV_GGUF`): discovery, served name, and a 64-option Choice over HTTP.
- `eval/tests/test_prompt_search.py`: GGUF flags and local-file/offline GGUF revision pinning.
- Removed the stale `tests/test_hf_backend.py`, which §4 already listed as removed; it imported the deleted `HFBackend` and only passed because it skipped without torch.

### 9.4 Validation (2026-09-24, Linux, Python 3.11, CPU, llama-cpp-python 0.3.35 built from source)

Hugging Face downloads were blocked in this environment, so real-engine checks used GGUF files built from llama.cpp's own vocabulary fixtures: the real Qwen2 vocabulary with the Qwen3.5-4B and Qwen2.5-7B chat templates, and the real Gemma 4 26B-A4B vocabulary with its embedded template, around small random-weight llama-architecture blocks. Tokenization and chat rendering are production-real; answers are not meaningful.

- `python -m pytest -c hf-server/pyproject.toml common/tests hf-server/tests -q`: 150 passed, 14 skipped (torch-only, Transformers-parity, and `SIMPLE_JEV_GGUF` tests); the 4 parity tests also pass with `transformers` installed.
- With `SIMPLE_JEV_GGUF` set to each of the three GGUFs: all 8 `test_llama_backend.py` tests pass, including shared-prefix vs independent full-prompt decode fidelity (with the Qwen3.5-template file, the whole `hf-server/tests` suite: 133 passed, 7 skipped).
- `python -m unittest discover -s eval -p 'test_*.py'`: 87 tests OK.
- Rendering parity: every branch compiled for all four formats, for state and chat requests including a 60-option Choice, rendered identically through `LlamaCppTokenizer` and Transformers 5.17's `render_jinja_template` (0 mismatches across the Qwen3.5, Qwen2.5, and Gemma 4 templates).
- End to end: the Qwen3.5-template GGUF served all four formats for a combined request with a 255-option Choice (255 probabilities returned). The Qwen2.5 template (no text-block content) and the Gemma 4 templates tried (llama.cpp's fixture, `google-gemma-4-31B-it.jinja`, and its `-interleaved` variant, which render reasoning only alongside tool calls) were rejected at startup for the named formats and served `baseline`.
- The real Gemma 4 26B-A4B GGUF header resolves to upstream's `Gemma MoE 26B-A4B` profile (`strict_mix_repeat2`).
- Live CLI server: `/health` and `/v1/models` report the served name; a request naming another model returned 422 under `--enforce-model-id`; chat under a named format returned upstream's 422; `eval/run.py` on SemIf-authored completed 144/144 rows and `eval/audit.py` verified 144 examples.

Not validated here: real-weight accuracy, GPU/Vulkan execution, and a full `prompt_search.py` run (the TypeSafe quick dataset requires preparation from a blocked host).

### 9.5 Follow-up: fidelity test with a hybrid model (Qwen3.5-0.8B-BF16)

User run on Windows (Python 3.14, Vulkan-enabled llama-cpp-python, RTX 4060),
`SIMPLE_JEV_GGUF=Qwen3.5-0.8B-BF16.gguf`: 132 passed, 7 skipped, 1 failed:
`test_shared_prefix_scores_match_independent_decodes` diverged by 0.028 logits
(tolerance 1e-3). Neither that test nor `LlamaCppBackend` changed in this merge,
and the test had previously been validated only on a dense model (§6).

First diagnosis (wrong): llama.cpp's default `op_offload` moves batches of 32+
tokens to Vulkan even with `n_gpu_layers=0`, which also disabled the fused
chunked Gated Delta Net kernel. That offload is real, so the test now disables it
under `SIMPLE_JEV_DEVICE=cpu`, but a pure-CPU rerun (no Vulkan compute buffer,
fused kernels enabled) still diverged: 0.0287 with prefix sharing and 0.0262 with
per-branch prefill.

Actual cause: recurrent/hybrid memory in llama.cpp slices a packed batch into
equal-length sub-batches. With the test's uneven suffixes (lengths 2, 3, 1), each
longer row's recurrence is computed in several pieces rather than in one pass as
in the single-sequence reference, which rounds differently. Reproduced on CPU
with random-weight `qwen35` hybrid GGUFs at realistic width (hidden 512, 8
layers): uneven rows drifted by 0.001 (F32) through the backend, and by up to
0.135 in full-vocabulary logits (BF16) when packed directly, while equal-length
rows matched within 1e-5 on both strategies. Small F32 models hid the effect.
Both prefill strategies score correctly; the drift is the same whether or not
the prefix is shared, and it predates this merge.

Changes (test and docs only; server numerics untouched, so results stay
comparable with the pre-merge server):
- The test checks both `--prefix-sharing` strategies and asserts the reported
  `prefill_strategy`.
- Attention-only models keep the strict 1e-3 check on uneven rows.
  Recurrent/hybrid models must match within 1e-3 on equal-length rows and within
  0.1 on uneven rows. Non-CPU devices use the documented GPU tolerance (0.15).
- With `SIMPLE_JEV_DEVICE=cpu` the test context sets `op_offload = False`.
- Mutation check: reading the wrong row's logits diverges by about 10 logits
  (dense) and 30 (hybrid), so every tolerance still detects real errors.
- `hf-server/README.md` precision notes describe both effects.

Real-weight validation after this change (user's Windows machine, Python 3.14,
Vulkan-enabled llama-cpp-python 0.3.35, `SIMPLE_JEV_DEVICE` unset = `cpu`,
`PYTHONUTF8=1`), `python -m pytest -c hf-server/pyproject.toml hf-server/tests -q`:
- `SIMPLE_JEV_GGUF=Qwen3.5-0.8B-BF16.gguf` (hybrid `qwen35`): 134 passed, 7 skipped.
- `SIMPLE_JEV_GGUF=qwen2.5-0.5b-instruct-fp16.gguf` (attention-only `qwen2`):
  134 passed, 7 skipped.

Prompt identity for the documented Qwen3.5-0.8B example was also re-checked:
compiled with the pre-merge (`0583a0c`) and merged servers, the token IDs,
answer-token IDs, labels, and `usage.input_tokens` are identical, and
Qwen3.5-0.8B (24 layers, hidden 1024) is not a profiled size, so omitting
`--classifier-prompt-policy` still resolves to `baseline`.

---

## 10. Upstream sync: image chat on llama.cpp, shared prompt formats, Noul wording

Date: 2026-09-27
Scope: merge of `featherless-ai/simple-jev` `main` at `dae340e` (14 commits past
the previous merge base `5686b21`). The GGUF / llama.cpp backend is kept, and
upstream's Transformers image support is re-implemented on llama.cpp's libmtmd.

### What came from upstream unchanged

- Prompt formats `shared_examples_binary`, `shared_repeat_state` and the
  experimental `universal_shared`, which keep system instructions uniform across
  question types so mixed requests keep sharing their context. Auto-selection
  now uses only `baseline`, `shared_examples_binary` and `shared_repeat_state`
  (`AUTO_TUNE_POLICIES`). The Qwen dense 4B profile, for example, now resolves
  to `shared_examples_binary` instead of `strict_mix_repeat2`. `eval/prompt_search.py`
  searches those three by default; `--all-formats` opts into all seven.
- Baseline Noul wording `Encode probability 0.1 as 1, 0.2 as 2, and so on
  through 0.9 as 9.` (upstream `c077d5d`). This fork had already adopted it in
  `8fd1164`, so there was no conflict. Compared with the wording the README
  example was recorded with, it adds 22 input tokens to the example request
  (796 → 818) and changes its Noul score. See "Regression check" below.
- `render_chat` (coalescing adjacent user turns for strict-alternation templates
  such as Gemma 3), image request validation, resize bounds
  (`--max-image-width/--max-image-height/--default-image-max-*`,
  `media_io_kwargs.image`), and the SSRF-safe public HTTP(S) downloader
  (`hf_media.py`).
- Website, demos (web DOOM, emotion camera), agent API guide, eval docs and
  `experiments/vision_validation/` (upstream's Transformers evidence, kept as a
  record).

### What was re-implemented for llama.cpp

Upstream's `hf_vision.py` feeds Transformers processors and forks
`past_key_values`. In this fork, `hf_vision.py` keeps upstream's validation half
verbatim and replaces the model half with an `MtmdVision` adapter over the
`llama_cpp.mtmd_cpp` bindings that ship with llama-cpp-python 0.3.35:

- `--mmproj` loads the GGUF vision projector. It accepts a path, a file next to
  the model, or a file in the `--model` HF repository. `mmproj*.gguf` files are
  excluded from model discovery. The projector path is resolved before weights
  load, so a bad name fails fast.
- Rendering: image blocks become text blocks holding the mtmd media marker
  (llama-server's convention); the GGUF chat template renders them. Image
  turns stay block lists (SmolVLM renders only lists); text-only turns are
  flattened as upstream does. Marker text supplied by a client is rejected.
- Tokenization: each image is passed through `mtmd_tokenize` on its own, and the
  text between markers is tokenized with the GGUF vocabulary. This reproduces
  `mtmd_tokenize` of the whole prompt, verified on real models, but never
  merges two adjacent same-size images into a Qwen-VL video frame pair. Image
  embeddings appear in `Branch.token_ids` as request-unique negative placeholder
  IDs (`Branch.media_spans` records them). The existing common-prefix logic and
  usage accounting (which counts expanded image tokens, as upstream does)
  therefore work unchanged.
- Execution: `LlamaCppBackend._score` generalizes the existing loop. The shared
  prefix may contain image spans and is never cut inside one. Each image chunk is
  encoded once per request and its embeddings are copied to request-local
  memory; `mtmd_helper_decode_image_chunk` decodes them with M-RoPE 2-D
  positions and model-specific non-causal attention. Row sequences receive the
  prefix via `llama_memory_seq_cp(0, row, 0, prefix_pos)`, where `prefix_pos`
  accounts for M-RoPE images taking `n_pos < n_tokens` positions. Image-bearing
  parts of a row are decoded per row; the text after each row's last image is
  packed exactly like text suffixes. Text-only requests take the same decode
  sequence as before this merge.
- Metrics: `multimodal_shared_prefix` / `multimodal_independent`,
  `vision_forwards` (projector encodes) and `image_decodes`.
- `LlamaCppTokenizer` now re-raises template `raise_exception()` errors as
  `jinja2.TemplateError`, as Transformers does, so `render_chat`'s Gemma 3
  alternation fallback works.
- Upstream's Transformers-only tests (`test_vision_groups.py`,
  `test_vision_integration.py`, `test_vision_processors.py`,
  `test_inline_processors.py`, and the tiny-torch-model half of `test_vision.py`)
  are replaced by `tests/test_llama_vision.py`.
- Dependencies: `Pillow>=10.4` (image decoding only; imported lazily). No torch,
  torchvision or Transformers.

### Measurements

Real-engine checks, CPU, llama-cpp-python 0.3.35 built from source (Linux x86-64,
4 threads). `test_llama_vision.py` with `SIMPLE_JEV_GGUF` / `SIMPLE_JEV_MMPROJ`:

| Model (projector) | Token layout = `mtmd_tokenize` | Single-row & same-partition scores vs mtmd reference | Packed requests (both strategies) | HTTP red circle |
|---|---|---|---|---|
| unsloth Qwen3.5-0.8B-BF16 (`mmproj-F16.gguf`), hybrid, M-RoPE | equal | within 1e-3 | within 0.2 probability | red / circle |
| ggml-org Qwen3-VL-2B-Instruct-Q8_0, M-RoPE | equal | 0.0 | within 0.2 probability | red / circle |
| ggml-org SmolVLM-256M-Instruct-Q8_0, tiled images | equal | within 1e-3 | within 0.2 probability | red / circle |
| ggml-org gemma-3-4b-it-Q4_K_M (`mmproj-model-f16.gguf`), non-causal image attention | equal | 0.0 | 0.0 (branch 0) | red / circle |

Gemma 3 ran a lighter one-image script instead of the full test: its CPU image encode
takes ~400 s here, and the full test re-encodes images for every mtmd reference. The
script's request has two adjacent user turns, so it also exercises the
strict-alternation fallback.

Finding: llama.cpp's CPU kernels are not batch-invariant. On Qwen3-VL-2B-Q8_0,
decoding a ~500-token branch's final token in its own batch moves its logits by
0.17, and packing branches together moves them by up to ~0.6. Splitting the
same text mid-way changes nothing. **Text-only** requests on the unchanged,
pre-merge code path show the same 0.16–0.44 differences against independent
full-prompt decodes, with flash attention on or off and with CPU weight repacking
on or off. It is an engine property, not an image-path error. It was invisible
before because `test_llama_backend.py`'s fidelity test uses ~13-token prompts. The
image test therefore checks strict equality where the decode partition matches
and bounds fully packed requests in probability space. The precision notes in
`hf-server/README.md` no longer claim bit-exact shared-prefix scores in general.

### Regression check (`Qwen3.5-0.8B-BF16`, `--device cpu --dtype float32 --classifier-prompt-policy baseline`)

The README example request (bicycle / Max) was served before and after the merge:

- Pre-merge HEAD `ff78aad` and this merge return **byte-identical** JSON:
  `color` red 0.99998069, `support` score 1.52382, `dog` noul 0.036213,
  `usage.input_tokens` 818.
- The README's recorded output (796 tokens, noul 0.010085, color 0.99998021)
  predates `8fd1164`'s Noul wording and was measured on an RTX 4060. The
  pre-wording commit `d5d6875` on this CPU gives 796 tokens, noul 0.010079 and
  color 0.99998069, matching it up to GPU/CPU rounding. The 22-token and Noul
  differences come from the wording change that both this fork and upstream
  adopted.
- The README comparison was re-recorded on CPU with the current wording: upstream's
  PyTorch server (`dae340e`, PyTorch 2.14 CPU, Transformers 5.17, `Qwen/Qwen3.5-0.8B`)
  and this server (`unsloth/Qwen3.5-0.8B-GGUF` / `Qwen3.5-0.8B-BF16.gguf`) both report
  818 input tokens. Noul is 0.03558 vs 0.03621 and score 1.52727 vs 1.52382.
