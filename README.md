# Simple Jev Project with llama.cpp and GGUF implementation

Use compatible open models from huggingface, for structured classification and scoring, without training a separate classifier head.

Send shared context and a set of questions. Simple Jev reads the model's next-token logits for each question and builds a JSON response containing choices, rubric scores, or truth/support judgments. The model does not generate a JSON completion: the server constructs the response from the scores.

The current implementation runs locally with llama.cpp (GGUF models) through `llama-cpp-python`, with Vulkan GPU acceleration. Shared request validation, versioned prompt instructions, and response scoring live in the plain Python `common/` folder so other inference implementations can use the same rules.

## Evaluation tooling

The [evaluation framework](eval/README.md) includes native JevBench scoring,
text/decision and vision suites, dataset preparation, resumable HTTP execution,
raw-response audits, and matched-metric comparisons. The client needs no GPU and
talks to any compatible endpoint, including this GGUF server.

```bash
python3 eval/run.py --preset quick --list  # 477 development/selection decisions
python3 eval/run.py --preset full --list  # frozen text and vision selections
python3 -m unittest discover -s eval -p 'test_*.py'  # offline checks
```

Prepare the selected datasets and start a compatible classifier endpoint before
execution. Large datasets, model weights and raw evaluation runs are not bundled;
see the [reproduction instructions](eval/README.md#portable-presets-and-full-reproduction).

## Running the GGUF Server

Use Python 3.12 or newer. The commands below use Python 3.13.

```bash
# Clone the repository and create an environment.
git clone https://github.com/terasut-num/simple-jev-llama-cpp.git
cd simple-jev-llama-cpp
python3.13 -m venv .venv
source .venv/bin/activate

# Build llama.cpp with the Vulkan backend for GPU acceleration + Administrator level
# (requires a C/C++ toolchain and the Vulkan SDK). Omit for CPU-only.
# Windows cmd: set CMAKE_ARGS=-DGGML_VULKAN=ON
export CMAKE_ARGS="-DGGML_VULKAN=ON"
python -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan
python -m pip install -r requirements.txt

# Install the server, including the shared common modules.
python -m pip install -e './hf-server'

# Start with a small GGUF model on CPU.
python hf-server/hf_server.py \
  --model Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  --gguf-file qwen2.5-0.5b-instruct-fp16.gguf \
  --device cpu --dtype float32 \
  --classifier-prompt-policy baseline \
  --max-model-len 4096 \
  --max-batch-size 4 --max-batch-tokens 4096

# Alternatively, offload every layer to any available GPU backend
# (Vulkan devices are used automatically when the wheel supports them).
python hf-server/hf_server.py \
  --model Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  --gguf-file qwen2.5-0.5b-instruct-fp16.gguf \
  --device auto --dtype bfloat16 \
  --max-model-len 8192 \
  --max-batch-size 4 --max-batch-tokens 8192

# Alternatively, enable image chat by also loading the model's vision projector
# (mmproj) from the same GGUF repository; see hf-server/VISION.md.
python hf-server/hf_server.py \
  --model unsloth/Qwen3.5-0.8B-GGUF \
  --gguf-file Qwen3.5-0.8B-BF16.gguf --mmproj mmproj-F16.gguf \
  --device cpu --dtype float32 \
  --classifier-prompt-policy baseline \
  --max-model-len 4096 \
  --max-batch-size 4 --max-batch-tokens 4096

# Alternatively, run Laya Typed Decisions with its native encoder backend.
# Stop the previous server first, or choose a different --port.
python -m pip install -e './hf-server[laya]'
USE_TF=0 python hf-server/hf_server.py \
  --backend laya \
  --model convaiinnovations/laya \
  --subfolder typed-decisions \
  --device cpu \
  --rope-factor 2 --max-model-len 2048
```

`--model` accepts a local `.gguf` file, a directory containing exactly one
`.gguf`, or a Hugging Face repository id. For repositories with multiple GGUF
variants, select one with `--gguf-file NAME.gguf`; only that file is downloaded
on first run. Vision projectors (`mmproj*.gguf`) are never selected as the model;
pass one with `--mmproj` to accept images. The public model ID in `/health`,
`GET /v1/models`, and responses is `--served-model-name` when given, otherwise the
`--model` value. For CPU-only builds of llama.cpp, omit `CMAKE_ARGS`.

The CPU example passes `--classifier-prompt-policy baseline` explicitly because
Qwen2.5-0.5B is not one of the [profiled architectures](#model-sizes-and-recommended-prompt-formats);
without a flag the server would still use `baseline`, after a startup warning.

The GPU example offloads all layers through llama.cpp's device enumeration:
with a Vulkan-enabled wheel, weights, KV cache, and flash-attention kernels run
on the GPU, with CPU fallback when no device is present. Allow memory for the
full model weights, KV cache, and inference buffers; sparse expert activation
does not mean only the active experts occupy memory. This is a launch example,
not a verified full-size benchmark. GPU kernels introduce small floating-point
differences relative to CPU; use `--device cpu --dtype float32` when scoring
must be numerically anchored.

CPU execution verifies deterministic llama.cpp scoring, but does not guarantee
bit-for-bit agreement with the original PyTorch server. `--dtype float32`
selects the llama.cpp KV-cache precision only. Different kernels, weight formats, 
and accumulation behavior can therefore shift raw logits and their softmax probabilities 
slightly even when the prompt, token accounting, and selected answer are the same.

The Laya example loads the specialized Typed Decisions checkpoint. Use `--device cuda` for an NVIDIA GPU, and send `"model": "convaiinnovations/laya"` in API requests. Its default native limit is 1,024 tokens per question. This example explicitly enables experimental 2× linear RoPE interpolation and a 2,048-token sequence budget, including instructions, options, and state. Both full and sliding attention rotary frequencies are halved; the local attention window is unchanged. This enables longer inputs but does not establish accuracy or calibration beyond the checkpoint's training length. Omit `--rope-factor 2` to retain the native behavior. See [Laya backend details](hf-server/README.md#laya-backend).

The server listens on `http://127.0.0.1:8000`. Once the model is loaded:

```bash
curl http://127.0.0.1:8000/health
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation. After installation, `simple-jev` and `python -m hf_server` accept the same arguments as the script.

## Set question, context, and Choice limits

These are **server startup flags**, not fields in a classification request:

| What to limit | Flag | Default | Meaning |
|---|---|---|---|
| Questions per HTTP request | `--max-request-branches` | `100` | Each question uses one scoring branch. Set `256` to allow the schema maximum; higher values cannot bypass that maximum. |
| Model input length | `--max-model-len` | `16384` | Maximum **tokens per complete rendered question branch**, including state/history, system instructions, options, template overhead, and any policy repetition—not characters or generated tokens. |
| Choices per Choice question | `--max-choice-options` | `255` | Configurable from `2` to `255`. This is per question, not the number of questions. Score remains limited to 50 levels; Noul is unchanged. |

For a Qwen 27B GGUF, allow up to 256 questions per request, 32K input tokens per
branch, and 255 choices per Choice question:

```bash
simple-jev --model /models/Qwen3.8-27B-Q4_K_M.gguf --device auto --dtype bfloat16 \
  --served-model-name qwen-27b \
  --max-request-branches 256 \
  --max-model-len 32768 \
  --max-choice-options 255 \
  --enforce-model-id
```

With no format flag, this recognized Qwen 27B configuration auto-selects
`examples_binary`, whatever the file is called. For example, a request with **3 questions containing 255 choices
each** uses 3 branches, not 765. Branch batch size is a separate throughput control;
you do not need to set `--max-batch-size` to the question limit.

These are independent upper bounds, not a guarantee that every combination fits.
Long descriptions and repeated-input policies consume more tokens. Over-limit
requests are rejected with **422**, never silently truncated. Raising input length
does not extend the checkpoint's native context support or available memory.
Use a supported context size and budget for extra memory/work: `--max-model-len`
also sizes the llama.cpp KV pool and is clamped to the GGUF's trained context. The
32K setting fits the checked 255-choice prompts, but arbitrarily long descriptions
may not. Above 50 options, Choice uses two-letter labels (`AA`, `AB`, …) that the
server checks, at startup, are single distinct tokens in the GGUF vocabulary;
a vocabulary with too few such labels fails to start until `--max-choice-options`
is lowered.
`max_tokens` in a request is not a substitute: this server scores without generating
output tokens. See [batch/token-budget details](hf-server/API_REFERENCE.md#limits-batching-and-cancellation).

## Model sizes and recommended prompt formats

When `--classifier-prompt-policy` is **omitted**, the GGUF loader reads the file's
GGUF header and matches the language-backbone architecture and size—not the file
name, repository name, or public alias—to these development-selected formats:

| Architecture / size (GGUF `general.architecture`) | Reference model | Auto-selected format |
|---|---|---|
| Qwen dense, 4B (`qwen35`) | `Qwen/Qwen3.5-4B` | `strict_mix_repeat2` |
| Qwen dense, 27B (`qwen35`) | `Qwen/Qwen3.8-27B` | `examples_binary` |
| Qwen MoE, 35B total / A3B active (`qwen35moe`) | `Qwen/Qwen3.6-35B-A3B` | `repeat_state` |
| Gemma unified dense, 12B (`gemma4`) | `google/gemma-4-12B-it` | `strict_mix_repeat2` |
| Gemma MoE, 26B total / A4B active (`gemma4`) | `google/gemma-4-26B-A4B-it` | `strict_mix_repeat2` |

The match checks backbone type, depth, widths, attention dimensions, vocabulary,
and expert configuration, translated from GGUF keys such as `block_count`,
`embedding_length`, `attention.head_count_kv`, `attention.key_length(_swa)`,
`feed_forward_length`, and `expert_*` (MTP layers appended by the converter are
not counted). Quantization does not change the match: a Q4_K_M and an F16 file of
the same checkpoint get the same format, although quantization can change which
format actually scores best. A family name or approximate parameter count alone
is not enough. Unknown configurations (including unregistered smaller sizes and
other architectures such as `qwen2` or `llama`) use `baseline` with a
**prominent startup warning to run prompt tuning first**. Laya keeps its native
format and is not part of this selection.

Named formats render through the GGUF's own chat template with text-block
content, a fixed `[thinking]` assistant prefill (`reasoning_content`), and
Transformers' `continue_final_message` rule, so prompts match upstream's
Transformers server.
The server compiles a sample request at startup: if the embedded template cannot
render the selected format—for example an older Gemma 4 template, or one that only
renders reasoning next to tool calls—it refuses to start and says so. Supply a
compatible template with `--chat-template-file`, or choose another format;
`baseline` works with any chat template.

An explicit flag always wins, including `--classifier-prompt-policy baseline`.
Use explicit `baseline` to reproduce the former no-flag behavior or to use
`messages`; the three named policies currently require text/JSON `state`.
Existing evaluation launches with an explicit policy keep that policy.
Recommendations are starting points, not universal optima for every fine-tune,
revision, quantization, or precision. See [format details](hf-server/README.md#prompt-format-selection).

### Search for the best format on quick eval

After [preparing the quick datasets](eval/README.md#portable-presets-and-full-reproduction)
and activating the GGUF-server environment:

```bash
# Inspect the four-policy plan without loading weights or downloading anything.
python eval/prompt_search.py --model /models/Qwen3.8-27B-Q4_K_M.gguf --list

# Stop any server using port 8179 first. Requires hardware for this checkpoint.
python eval/prompt_search.py \
  --model /models/Qwen3.8-27B-Q4_K_M.gguf \
  --device auto --dtype bfloat16 --max-model-len 32768 \
  --output eval/results/qwen27b-prompt-search

# A Hugging Face GGUF repository: pin the commit and select one file.
python eval/prompt_search.py \
  --model ORG/MODEL-GGUF --gguf-file MODEL-Q4_K_M.gguf --revision COMMIT_SHA \
  --device auto --max-model-len 32768 --output eval/results/my-gguf-search
```

The tool starts and stops its own local GGUF server sequentially for `baseline`,
`examples_binary`, `repeat_state`, and `strict_mix_repeat2`: **477 cases per
format, 1,908 total**. It pins remote revisions (local GGUF files are recorded by
path and size and must stay unchanged during the search), preserves logs/raw
responses and native summaries, and writes `comparison.json`. `--gguf-file`,
`--n-gpu-layers`, `--prefix-sharing`, and `--chat-template-file` are passed to
every run unchanged. Incomplete searches do not get a
recommended winner. Selection uses pooled native correct/477; ties use requested
policy order (baseline first by default). Apply the reported format explicitly
with `--classifier-prompt-policy`; the tool never rewrites defaults or this table.

Quick is a **development/selection set**, not held-out accuracy evidence. Validate
on disjoint data before claiming generalization. This search is not a 255-option
accuracy benchmark. See [search controls and artifacts](eval/README.md#prompt-format-search).

## How do I use the API?

Send a non-streaming `POST /v1/classifier` request. With `--enforce-model-id`, the `model` value must exactly match the served ID; otherwise any value is accepted and the response reports the served ID. The examples below use the Qwen2.5 GGUF started above. Supply exactly one of:

- `state`: a string, JSON object, or JSON array containing the shared context.
- `messages`: chat history, rendered using the model's own chat template. User turns may include `image_url` blocks when the server loads a vision projector (`--mmproj`); see [image chat inputs](hf-server/VISION.md).

The API takes inspiration from TypeSafe's structured-decision interface and includes project-specific behavior. `/v1/systemone` is an alias of `/v1/classifier`; both run the same implementation. Use this repository's [API reference](hf-server/API_REFERENCE.md) as the contract for clients. `GET /v1/models` advertises the served ID (`--served-model-name`, defaulting to `--model`) and the configured Choice cap as `x_max_choice_options`; it never runs inference. Request model IDs are unchecked unless `--enforce-model-id` is set. Choice defaults to a 255-option cap; set `--max-choice-options` to lower it. Formats for 50 or fewer choices are unchanged. The three legacy named prompt formats (`examples_binary`, `repeat_state`, `strict_mix_repeat2`) accept `state` only; chat `messages` need `baseline` or a `shared_*`/`universal_shared` format.

```bash
curl http://127.0.0.1:8000/v1/classifier \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "model": "unsloth/Qwen3.5-0.8B-GGUF",
  "state": "Mia owns a red bicycle. Her dog is named Max.",
  "questions": {
    "color": {
      "type": "choice",
      "instructions": "What color is Mia's bicycle?",
      "criteria": {"red": null, "blue": null}
    },
    "support": {
      "type": "score",
      "instructions": "How well does the context support that Mia owns a bicycle?",
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

Question IDs become keys in `answers`.

### CPU comparison with the original PyTorch server from [`Original repository`](https://github.com/featherless-ai/simple-jev)

The following results use the same request. The original server ran the
Hugging Face `Qwen/Qwen3.5-0.8B` checkpoint with PyTorch and A100 (80 GB):

```json
{
    "model": "Qwen/Qwen3.5-0.8B",
    "answers": {
        "color": {
            "type": "choice",
            "confidence": 0.9999804496765137,
            "probabilities": {
                "red": 0.9999804496765137,
                "blue": 1.9588253053370863e-05
            },
            "choice": "red"
        },
        "support": {
            "type": "score",
            "confidence": 0.5364056825637817,
            "probabilities": {
                "0": 0.009136191569268703,
                "1": 0.4544581472873688,
                "2": 0.5364056825637817
            },
            "score": 1.5272694826126099,
            "legend": {
                "0": "Unsupported",
                "1": "Partially supported",
                "2": "Fully supported"
            }
        },
        "dog": {
            "type": "noul",
            "noul": 0.010081952810287469
        }
    },
    "usage": {
        "input_tokens": 796,
        "output_tokens": 0
    }
}
```

The llama.cpp server ran `Qwen3.5-0.8B-BF16.gguf` on GPU (RTX 4060) with `--dtype float32 --classifier-prompt-policy baseline`:

```json
{
  "model": "unsloth/Qwen3.5-0.8B-GGUF",
  "answers": {
    "color": {
      "type": "choice",
      "confidence": 0.9999802112579346,
      "probabilities": {
        "red": 0.9999802112579346,
        "blue": 1.9806906493613496e-05
      },
      "choice": "red"
    },
    "support": {
      "type": "score",
      "confidence": 0.5352787971496582,
      "probabilities": {
        "0": 0.009363764896988869,
        "1": 0.45535746216773987,
        "2": 0.5352787971496582
      },
      "score": 1.5259150266647339,
      "legend": {
        "0": "Unsupported",
        "1": "Partially supported",
        "2": "Fully supported"
      }
    },
    "dog": {
      "type": "noul",
      "noul": 0.010085486769676195
    }
  },
  "usage": {
    "input_tokens": 796,
    "output_tokens": 0
  }
}
```

Both servers selected `red`, placed `support` toward the third rubric level,
and reported 796 input tokens. Their probabilities and score differ slightly,
as expected from the FP16 GGUF weights and differing PyTorch/llama.cpp numeric
implementations described above.

| Question type | Input criteria | Result |
| --- | --- | --- |
| `choice` | Object with 2–255 candidate IDs and optional descriptions, subject to `--max-choice-options` | Highest-probability candidate, its confidence, and the candidate distribution. |
| `score` | Array of 2–50 rubric levels, lowest to highest | Expected zero-based rubric index, confidence, distribution, and rubric legend. A three-level rubric returns a value from 0 to 2, including fractional values. |
| `noul` | Optional `true` and/or `false` descriptions | Truth/support judgment from 0.01 to 0.99, derived from the model's distribution over nine rating tokens. |

Choice and score confidence is the largest probability among their allowed labels. These distributions, and the Noul value, are not calibrated probabilities of correctness.

For chat input, send `messages` instead of `state` (with the `baseline` format). This complete example classifies a customer message and provides explicit descriptions for the possible answers:

```bash
curl http://127.0.0.1:8000/v1/classifier \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "model": "Qwen/Qwen3.5-0.8B",
  "messages": [
    {"role": "user", "content": "I was charged twice for my subscription. Please refund the duplicate charge."}
  ],
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should handle this message?",
      "criteria": {
        "billing": "Payments, invoices, and refunds",
        "technical": "Errors and problems using the product"
      }
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "Does the customer explicitly request a refund?",
      "criteria": {
        "true": "The customer asks for money back.",
        "false": "The customer makes no refund request."
      }
    }
  }
}
JSON
```

Unknown top-level request fields are ignored, including completion settings such as `temperature`, `max_tokens`, and `stream`. Unknown fields inside questions and options are rejected. There is no completion sampling or streaming. The GGUF server supports text and [image chat inputs](hf-server/VISION.md) when started with the model's vision projector (`--mmproj`, e.g. `mmproj-F16.gguf`), through llama.cpp's libmtmd: Qwen-VL/Qwen3.5, Gemma 3, SmolVLM and other projector families llama.cpp supports. Each image is encoded once per request. Images use `image_url` blocks in user `messages`, with base64 data URLs or bounded public HTTP(S) downloads. Private-network URLs, audio, video, and tool calls are rejected.

`usage.input_tokens` counts unique token prefixes within the request, sharing the common context across questions. `usage.output_tokens` is zero because no output tokens are generated. For diagnostic timings, start the server with `ENABLE_OPEN_JEV_ADVANCED_METRICS=1`; adding `"options": {"raw_logits": true}` to a request then includes selected-token logits.

## What is a classifier, and why “System One”?

A classifier maps input to a defined set of answers. For example, a support system might route a message to `billing` or `technical`, score its urgency against an ordered rubric, and judge whether it requests a refund. Those decisions can feed directly into ordinary application code.

TypeSafe uses “System One” to describe models designed for fast, structured decisions, drawing the name from the distinction between fast intuitive thinking and slower deliberate reasoning. Its [introduction to System One and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) explains that motivation. Simple Jev explores this style of interface using existing open language models. It does not reproduce TypeSafe's model architecture or training, or establish equivalent accuracy, calibration, or speed.

In this implementation, the useful change is how the model is used:

1. The shared prompt builder creates consistent classifier instructions and one scoring branch per question.
2. The server renders those instructions and the context into the model's native chat format from the GGUF's chat template.
3. It decodes the exact common token prefix once and copies those KV cells into per-question sequences for each suffix batch.
4. It reads the next-token logits for the allowed answer labels. Shared scoring code normalizes those scores and constructs the JSON response.

This avoids generating and parsing a prose or JSON answer token by token. Reusing the context can also reduce repeated computation when several questions refer to the same input. Actual latency depends on the model, hardware, context length, and number of questions. Cache reuse currently lasts only for a single request.

A valid response structure does not guarantee a correct decision. Model capability and question wording still matter; evaluate answer quality on your own task separately from validating the server's inference and response pipeline.

## How shared prompts and prefill-only scoring reduce work

A normal text-generation request has a **prefill** step that processes the input, followed by **decode** steps that generate tokens one at a time. Prefill already produces logits for the next token. Simple Jev uses those logits directly to score predefined answer labels, so it needs no autoregressive decode loop.

For several questions about the same context, most of the prompt is identical. After applying the model's chat template and tokenizing each question's prompt, the server finds their exact common token prefix:

```text
Shared instructions + context + shared question briefing
                           │
                     Prefill once
                     Save KV cache
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
  Color question    Support question    Dog question
         │                 │                 │
    Label logits      Label logits      Label logits
         └─────────────────┼─────────────────┘
                           ▼
                JSON built by the server
```

The KV pool stores the model's attention state for the shared prefix. Each question continues from a copy of those cells with its own suffix and answer prefix. The server packs these suffixes into batched decodes, reads the logits at each row's last real token, and passes the selected label scores to the shared response scorer. Questions do not consume one another's answers.

For example, suppose four question prompts each contain a 1,000-token common prefix and a 50-token suffix:

| Execution | Prompt tokens processed, excluding padding |
| --- | --- |
| Evaluate each complete prompt separately | `4 × (1,000 + 50) = 4,200` |
| Reuse the shared prefix | `1,000 + 4 × 50 = 1,200` |

If all four suffixes fit in one batch, the shared execution takes one prefix decode and one batched suffix decode. These token counts illustrate avoided repeated input processing, not a measured latency ratio: each suffix still attends to the cached prefix, and KV copies and model execution have costs.

`--max-batch-size` limits questions per suffix batch; `--max-batch-tokens` limits the number of packed suffix tokens in that batch. Neither limits total model/cache memory or chunks the shared prefix. The current server reuses the KV pool within a request and processes model requests serially. See the [GGUF execution guide](hf-server/README.md#shared-prefix-execution) for details.

Sharing the prefix depends on the model being able to copy its cached state between sequences. `--prefix-sharing auto` (the default) does that for ordinary attention models, and falls back to prefilling each question's full prompt for recurrent or hybrid architectures, which carry rolling state instead of per-position cells. Those models used to be rejected at startup; `--prefix-sharing on` opts them back into sharing. See [Recurrent and hybrid architectures](hf-server/README.md#recurrent-and-hybrid-architectures).

## Shared prompt contract and project layout

| Location | Purpose |
| --- | --- |
| [`common/`](common/README.md) | Plain Python modules for `ClassifierRequest`, prompt planning, and response scoring. No separate package installation is required. |
| [`common/PROMPT_STRUCTURE_V1.md`](common/PROMPT_STRUCTURE_V1.md) | Language-independent v1 specification: inputs, prompt strings, chat roles, answer labels, and scoring rules. |
| [`hf-server/hf_server.py`](hf-server/hf_server.py) | Single-file llama.cpp implementation: GGUF loading and header fingerprinting, chat rendering, shared-prefix inference, HTTP API, and CLI. |
| [`hf-server/hf_prompt_policies.py`](hf-server/hf_prompt_policies.py) | Upstream's startup-selected prompt formats and architecture/size recommendations, used unchanged by the GGUF server. |
| [`hf-server/API_REFERENCE.md`](hf-server/API_REFERENCE.md) | Detailed request/response contract, validation, diagnostics, and configuration. |
| [`eval/`](eval/README.md) | Endpoint evaluation suites, presets, audits, and the GGUF prompt-format search. |
| [`CHANGED.md`](CHANGED.md) | Log of this fork's llama.cpp/GGUF changes and upstream synchronizations. |
| [`RFDT/`](RFDT/README.md) | Task-specific decision training: prepare labels, distill teacher estimates, train on answer-token logits, and export a student. |

`prepare_prompt(request, version="v1")` returns a cacheable system prompt prefix, prefix instruction, suffix instruction, and ordered questions. The inference implementation handles chat formatting and model execution; `common/response_scoring.py` converts label logits or mapped PyTorch tensors into answers.

The version fixes one prompt/scoring configuration so implementations can stay consistent, including implementations in other languages. It defaults to `v1`; the HTTP API currently uses that version. There are no per-request independent/rating modes or score-format switches.

## Testing

From the repository root, with the environment activated:

```bash
python -m pip install -e './hf-server[test]'
python -m pytest -c hf-server/pyproject.toml common/tests hf-server/tests -q
```

The tests cover request validation, prompt construction, prompt formats, model discovery and served names, 255-option Choice labels, GGUF header fingerprinting, chat-template continuation, response scoring, HTTP behavior, and shared-prefix-versus-full-prompt inference. The GGUF backend's orchestration and loader tests run against a stubbed engine without any weights; its real-engine tests are opt-in via `SIMPLE_JEV_GGUF` pointing at a local GGUF file (`SIMPLE_JEV_DEVICE` defaults to `cpu`). If `transformers` happens to be installed, extra tests check that chat rendering matches Transformers exactly; it is not a runtime dependency. The tests do not measure classification accuracy.

```bash
SIMPLE_JEV_GGUF=/models/model.gguf python -m pytest -c hf-server/pyproject.toml hf-server/tests -q
```

Models need a GGUF file with a usable chat template (embedded, or supplied with `--chat-template-file`) and answer labels that each extend the rendered prompt by exactly one distinct token. Architectures whose KV state cannot be shared across sequences — recurrent and hybrid stacks — are supported too, prefilling each question separately instead. The server checks label tokenization; compatibility with every open model is not guaranteed.

## One more thing: Really Fancy Decision Training (RFDT)

Want a smaller model that is better at your specific use case? **RFDT** lets you fine-tune a model on the decisions your application needs. Provide context or chat history, questions, and answers—or let a larger teacher model supply the missing answers.

RFDT trains directly on the allowed answer-token logits using the same prompt structure as Simple Jev inference. The scripts support dataset preparation, teacher labeling, multi-GPU training, LoRA adapters, evaluation, and export of a Hugging Face checkpoint; convert it with llama.cpp's `convert_hf_to_gguf.py` to serve it here. See the [RFDT guide and examples](RFDT/README.md) to get started on your own hardware.

As we scale up support and usage of Simple Jev models on [Featherless](https://featherless.ai/), we will roll out support for serving fine-tuned models and running fine-tuning on the platform. The RFDT scripts are available in this repository today; hosted fine-tuned model support and fine-tuning are part of that upcoming rollout.
