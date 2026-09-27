# Image chat inputs

The llama.cpp backend accepts images in chat `messages` when the server is
started with the GGUF **vision projector** (`mmproj`) that matches the text
model. Images run through llama.cpp's multimodal library, libmtmd, bundled with
`llama-cpp-python`. No PyTorch, Transformers or Hugging Face processor is used.

```bash
python hf-server/hf_server.py --model unsloth/Qwen3.5-0.8B-GGUF \
  --gguf-file Qwen3.5-0.8B-BF16.gguf --mmproj mmproj-F16.gguf --device auto
```

`--mmproj` accepts a local path, a file name next to the resolved model GGUF (or
inside a local `--model` directory), or a file name in the `--model` Hugging Face
repository, which is then downloaded on its own. Projector files (`mmproj*.gguf`)
are never mistaken for the text model when a directory or repository is scanned.
The projector runs on the GPU when any text layer is offloaded (`--device auto`),
and on the CPU with `--device cpu`. Without `--mmproj`, image requests return HTTP
422 and text requests are unaffected. Advanced metadata reports `image_input` and
`mmproj_path`.

Any projector family llama.cpp supports can be used. It was exercised with
Qwen3.5 (hybrid, M-RoPE), Qwen3-VL (M-RoPE), Gemma 3 (non-causal image
attention) and SmolVLM/Idefics3 (tiled images); see
[Validation scope](#validation-scope). GGUF repositories usually ship the
projector beside the weights, e.g. `mmproj-F16.gguf` in `unsloth/Qwen3.5-0.8B-GGUF`
or `mmproj-model-f16.gguf` in `ggml-org/gemma-3-4b-it-GGUF`.

Use `baseline`, `shared_examples_binary`, `shared_repeat_state`, or
`universal_shared` with chat. The legacy state-only policies remain state-only.
Existing automatic policy recommendations were selected on text tasks; they are
not newly established vision-quality recommendations. `shared_*` Choice prompts
require a template that preserves `reasoning_content`; non-thinking templates
(e.g. Qwen2.5-VL/Qwen3-VL Instruct) are rejected at startup for those formats. Use
`baseline` for those checkpoints (or explicitly experiment with `universal_shared`).
No image input is inferred from JSON `state`.

## Request

Each user message may mix `text` and `image_url` blocks. Multiple images and
multiple conversation turns retain their original order. When a native template
requires strict user/assistant alternation (e.g. Gemma 3), adjacent user content
is coalesced without dropping image/text blocks so the appended classifier
question can be rendered. Other templates retain the separate turns.
For example, after starting a model with its projector on port 8000:

```python
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

image = base64.b64encode(Path("photo.png").read_bytes()).decode("ascii")
payload = {
    "model": "unsloth/Qwen3.5-0.8B-GGUF",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Inspect this picture."},
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + image
        }},
    ]}],
    "questions": {
        "animal": {"type": "choice", "instructions": "Which animal is shown?",
                   "criteria": {"cat": None, "dog": None, "other": None}},
        "outside": {"type": "noul", "instructions": "Is the scene outdoors?"},
    },
}
request = Request("http://127.0.0.1:8000/v1/classifier",
                  data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
with urlopen(request) as response:
    print(json.load(response))
```

`/v1/systemone` accepts the identical payload. Responses use the existing
Choice/Score/Noul schema; the server scores logits, not generated descriptions.
A public image URL can replace the data URL without changing the request shape:

```json
{"type":"image_url","image_url":{"url":"https://example.com/photo.jpg"}}
```

Images are inputs only; the server does not generate images.

## How images reach the model

1. Images are validated and decoded with Pillow (EXIF orientation, RGB, the
   resize bounds below). This is the upstream server's code, unchanged.
2. Following llama-server's convention, every image block becomes a text block
   holding libmtmd's media marker (`<__media__>`), and the conversation is
   rendered with the GGUF's own chat template. Image turns stay block lists, so
   templates that only render list content (e.g. SmolVLM) keep them. A request
   whose text already contains the marker is rejected rather than misread.
3. Each image is tokenized **on its own** with `mtmd_tokenize`, which applies the
   projector's preprocessing and the model's image begin/end tokens (e.g. Qwen's
   `<|vision_start|>`/`<|vision_end|>`, Gemma's `<start_of_image>`). A branch's
   tokens are then the GGUF tokenization of each text span between markers plus
   those image layouts, exactly what `mtmd_tokenize` produces for the complete
   prompt (checked on real models, below). Tokenizing images individually
   keeps every image its own span: llama.cpp would otherwise merge two adjacent
   same-size images into one Qwen-VL video-frame pair.
4. In compiled branches, image embeddings appear as request-unique negative
   placeholder IDs. Common-prefix detection and usage accounting therefore see
   expanded image tokens without inventing vocabulary IDs, and equal prefixes
   imply equal images.
5. Inside the backend lock, each image chunk is encoded by the projector **once
   per request** and its embeddings are copied into request-local memory. They
   are decoded into the KV cache with `mtmd_helper_decode_image_chunk`, which sets
   M-RoPE 2-D positions and, where the model needs it, non-causal attention inside
   the span. Text after an M-RoPE image continues at `n_past + n_pos`, as in
   llama.cpp's own helpers.

## Image resize configuration

Resize oversized images **to fit** a bounding box, preserving aspect ratio,
without cropping or upscaling smaller images. These are resize settings, not
rejection thresholds. Three levels are supported:

```bash
simple-jev --model YOUR_MODEL.gguf --mmproj mmproj-F16.gguf \
  --max-image-width 1920 --max-image-height 1080 \
  --default-image-max-width 1024 --default-image-max-height 768
```

- `--max-image-width` / `--max-image-height`: server hard caps on the decoded
  image passed to the projector. A request cannot increase these caps.
- `--default-image-max-width` / `--default-image-max-height`: default request
  bounds. Defaults must not exceed corresponding hard caps.
- Per-request `media_io_kwargs.image.max_width` / `max_height`: override each
  default independently, clamped to the corresponding server hard cap:

```json
"media_io_kwargs": {"image": {"max_width": 1600, "max_height": 900}}
```

Here the request uses a 1600×900 box rather than the default 1024×768. Requesting
4096×2160 instead uses the hard 1920×1080 box. A 4000×3000 image in that box becomes
1440×1080. Omitted request axes retain their defaults. An unset server default
falls back to its hard cap; an unset hard cap imposes no bound on that axis.
All four startup options default to unset, preserving previous behavior.
Dimensions must be positive integers; null/zero are not ways to bypass a cap.

This applies identically to base64 and URL images, after EXIF orientation/RGB
conversion and before the projector's preprocessing, once per image occurrence
per request. Original byte/pixel/animation safety checks still apply **before**
resizing. It does not reduce upload size or avoid initial image decoding. The
projector may subsequently resize, tile, upscale or pad to its required grid;
these are **projector-input caps**, not guarantees of final image-token counts,
memory or speed. Smaller images can lose OCR/fine detail. Laya remains text-only
and rejects these options.

## Sharing and correctness

- Decode, preprocess and encode each image once per request, however many
  questions the request asks.
- Reuse follows the **actual expanded token prefix**, not the policy name. When
  every image span lies in the common prefix (the usual case: images in the chat,
  the question appended after), the prefix, including image embeddings, is
  decoded once on sequence 0 and copied to each question's sequence with
  `llama_memory_seq_cp` (`multimodal_shared_prefix`). The shared prefix is never
  cut inside an image span.
- Recurrent/hybrid models (e.g. Qwen3.5) default to per-branch prefill, as for
  text (`--prefix-sharing auto`). Each branch then decodes the once-encoded image
  embeddings on its own sequence (`multimodal_independent`); the projector
  still runs once per image. `--prefix-sharing on|off` overrides this.
- Text after the last image of each branch is packed with the other branches
  into one decode, as for text requests. `--max-batch-tokens` bounds each
  branch's suffix, counting image tokens beyond the shared prefix.
- The complete **expanded** prompt, including image tokens, must fit
  `--max-model-len`; no truncation is performed. Public input usage is the token
  prefix union, including expanded image tokens. Output accounting is unchanged.
  Advanced metrics report `vision_forwards` (projector encodes) and
  `image_decodes` in addition to the text metrics.
- Media and embeddings are request-local, never a cross-request image cache.
  Every request starts from an empty KV pool.

## Current limits

- PNG, JPEG, and WebP, via inline base64 data URLs or public HTTP(S) URLs.
  Camera EXIF orientation is honored. Local filesystem paths are never opened.
- Remote URLs use standard ports (HTTP 80 / HTTPS 443), no URL userinfo, ambient
  authentication, cookies or proxies. Every DNS address must be public; the validated IP is pinned
  to the socket while preserving the original Host/TLS hostname. Each redirect
  is checked again. Loopback, private, link-local, reserved and multicast
  addresses are rejected, including private IPv4 transition encodings.
- Downloads have a 10-second wall-time limit each and a 20-second request media
  budget, at most three redirects, and bounded streaming even without a content
  length. DNS concurrency is bounded. HTTP compression is not accepted. Network
  failures return HTTP 422; there is no insecure TLS or private-network fallback.
- Up to 16 images; 10 MiB encoded-image bytes each, 20 MiB total; 20 million
  decoded pixels each, 40 million total. Animated images are rejected.
- `image_url.detail` may be omitted or `"auto"`. Unsupported block fields,
  audio/video, tools, nonempty `mm_processor_kwargs`, and `media_io_kwargs`
  other than the image resize keys above return 422 rather than being silently
  ignored. Audio projectors are not enabled.
- Images are accepted only in user messages. Other roles can contain text
  strings or text blocks, subject to the model's native chat-template rules.
- Projectors whose image attention is non-causal (e.g. Gemma 3) need each image
  span within one micro-batch; llama.cpp's default `n_ubatch` (512) covers
  Gemma 3's 256 image tokens. A decode failure surfaces as an error, never as a
  silently truncated image.
- Laya remains text-only.

## Numerical behavior

llama.cpp's CPU kernels are not batch-invariant: identical tokens decoded in a
batch of a different size or composition (a one-token tail, several packed
question rows) round differently. This is a property of the engine, the same
for text-only requests and for llama-server. Measured on CPU for ~500-token
branches, up to ~0.6 logits for Qwen3-VL-2B-Q8_0, and up to ~0.1 for
Qwen3.5-0.8B-BF16 with its hybrid recurrence. When the decode partition is the
same, the backend is **exact**: every branch scored alone equals llama.cpp's own
`mtmd_helper_eval_chunks` evaluation of the complete prompt with zero difference
on CPU, and the shared-prefix path equals a single-sequence decode with the same
prefix/suffix split. Packing therefore changes rounding, not which tokens,
images or positions the model sees. GPU backends add their own kernel
differences. None of this establishes vision quality or calibrated
probabilities.

## Validation scope

Offline tests (`tests/test_llama_vision.py`, no weights) use real layout objects
with a stub projector and engine. They cover marker rendering, per-image spans
and placeholder IDs, label boundaries after the final image, expanded-token
limits, resize bounds, M-RoPE positions for prefix and suffix decodes, sequence
copies, once-per-request encoding, both prefill strategies, cancellation,
loading, projector resolution, CLI wiring and HTTP 422 without a projector. The
upstream validation, resize and SSRF-safe download tests (`test_vision.py`,
`test_image_resize.py`, `test_media.py`) are kept unchanged.

Real-engine tests run when `SIMPLE_JEV_GGUF` and `SIMPLE_JEV_MMPROJ` name a
local GGUF and its projector:

```bash
SIMPLE_JEV_GGUF=/models/Qwen3.5-0.8B-BF16.gguf \
SIMPLE_JEV_MMPROJ=/models/mmproj-F16.gguf \
  python -m pytest -c hf-server/pyproject.toml hf-server/tests/test_llama_vision.py
```

For a two-image, multi-turn request they check that each branch's token layout
equals `mtmd_tokenize` of its complete rendered prompt. They check the strict
equalities above (1e-3 logits on CPU) and bound fully packed requests in both
prefill strategies to 0.2 absolute label probability. An HTTP test with a
generated red-circle image requires `red` and `circle` through both routes, plus
exact usage accounting. On CPU these tests passed for Qwen3.5-0.8B-BF16 with
`mmproj-F16.gguf`, Qwen3-VL-2B-Instruct-Q8_0 and SmolVLM-256M-Instruct-Q8_0
(ggml-org). Gemma 3 4B-it Q4_K_M was checked with a lighter one-image script
(same layout, exact single-row and packed agreement, `red`/`circle`, and the
strict-alternation fallback); see [CHANGED.md §10](../CHANGED.md). This is not a broad vision-quality, throughput, or GPU validation claim.
