"""Image request validation and the llama.cpp (libmtmd) multimodal adapter.

Remote images use a bounded public-network-only downloader; local paths are never
opened. Media is decoded on the CPU with Pillow, then handed to llama.cpp's
libmtmd through the GGUF vision projector (mmproj) that matches the text model.
Projector encodes and embedding decodes run only under LlamaCppBackend's lock.

The request-validation half of this module (limits, resize bounds, decoding)
matches the upstream Transformers server; the adapter half replaces its native
processor and KV-cache fork with mtmd tokenization, encoding and decoding.
"""
import base64
import binascii
import ctypes
import io
import logging
import time
import warnings
import weakref
from dataclasses import dataclass

MAX_IMAGES = 16
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_TOTAL_PIXELS = 40_000_000
MAX_MEDIA_SECONDS = 20


def validate_image_dimensions(max_image_width=None, max_image_height=None):
    """Optional preprocessor input bounds; never relax encoded/source-image limits."""
    for name, value in [('max_image_width', max_image_width), ('max_image_height', max_image_height)]:
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f'{name} must be a positive integer or None')


def validate_image_resize_config(max_image_width=None, max_image_height=None,
                                 default_image_max_width=None, default_image_max_height=None):
    validate_image_dimensions(max_image_width, max_image_height)
    for name, default, cap in [('default_image_max_width', default_image_max_width, max_image_width),
                               ('default_image_max_height', default_image_max_height, max_image_height)]:
        if default is not None and (type(default) is not int or default <= 0):
            raise ValueError(f'{name} must be a positive integer or None')
        if default is not None and cap is not None and default > cap:
            raise ValueError(f'{name} must not exceed the server hard cap')


def image_resize_bounds(media_io_kwargs, *, max_image_width=None, max_image_height=None,
                        default_image_max_width=None, default_image_max_height=None):
    """Resolve request overrides without mutating shared compiler configuration."""
    validate_image_resize_config(max_image_width, max_image_height,
                                 default_image_max_width, default_image_max_height)
    options = {} if media_io_kwargs is None else media_io_kwargs
    if not isinstance(options, dict) or set(options) - {'image'}:
        raise ValueError('media_io_kwargs supports only image max_width/max_height')
    image = options.get('image', {})
    if not isinstance(image, dict) or set(image) - {'max_width', 'max_height'}:
        raise ValueError('media_io_kwargs.image supports only max_width/max_height')
    for name, value in image.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f'media_io_kwargs.image.{name} must be a positive integer')
    bounds = []
    for key, default, cap in [('max_width', default_image_max_width, max_image_width),
                              ('max_height', default_image_max_height, max_image_height)]:
        value = image.get(key, default)
        bounds.append(cap if value is None else value if cap is None else min(value, cap))
    return tuple(bounds)


def image_messages(messages, *, max_image_width=None, max_image_height=None):
    """Copy OpenAI text/image_url blocks to image placeholder blocks and decoded PIL images.

    Images are RGB, in conversation order. Public HTTP(S) and inline data URLs
    are supported. Private networks, paths, video/audio, animation, unknown block
    options, and oversized payloads fail closed.
    """
    from PIL import Image, ImageOps

    validate_image_dimensions(max_image_width, max_image_height)
    result, images, total, total_pixels = [], [], 0, 0
    deadline = time.monotonic() + MAX_MEDIA_SECONDS
    for message in messages:
        item = dict(message)
        content = item['content']
        if isinstance(content, str):
            result.append(item)
            continue
        if not isinstance(content, list) or not content:
            raise ValueError('Vision chat content must be text or nonempty text/image_url blocks')
        blocks = []
        for block in content:
            if not isinstance(block, dict):
                raise ValueError('Content blocks must be objects')
            if block.get('type') == 'text' and set(block) == {'type', 'text'} and isinstance(block['text'], str):
                blocks.append(dict(block))
                continue
            if block.get('type') != 'image_url' or set(block) != {'type', 'image_url'}:
                raise ValueError('Only text and image_url content blocks are supported')
            if item['role'] != 'user':
                raise ValueError('Images are supported only in user messages')
            spec = block['image_url']
            if not isinstance(spec, dict) or set(spec) - {'url', 'detail'} or spec.get('detail', 'auto') != 'auto':
                raise ValueError('image_url requires url and optional detail=auto')
            url = spec.get('url')
            if not isinstance(url, str):
                raise ValueError('image_url.url must be a string')
            if len(images) >= MAX_IMAGES:
                raise ValueError('Image count limit exceeded')
            if url[:8].lower().startswith(('https://', 'http://')):
                from hf_media import fetch_image
                data = fetch_image(url, max_bytes=min(MAX_IMAGE_BYTES, MAX_TOTAL_BYTES - total),
                                   timeout=min(10, deadline - time.monotonic()))
            else:
                if ',' not in url:
                    raise ValueError('Images require HTTP(S) or base64 data URLs')
                header, payload = url.split(',', 1)
                if header.lower() not in {'data:image/png;base64', 'data:image/jpeg;base64', 'data:image/webp;base64'}:
                    raise ValueError('Images require PNG, JPEG, or WebP data URLs')
                if len(payload) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
                    raise ValueError('Image byte limit exceeded')
                try:
                    data = base64.b64decode(payload, validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise ValueError('Invalid image base64') from exc
            total += len(data)
            if len(data) > MAX_IMAGE_BYTES or total > MAX_TOTAL_BYTES:
                raise ValueError('Image byte limit exceeded')
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', Image.DecompressionBombWarning)
                    with Image.open(io.BytesIO(data)) as image:
                        if image.format not in {'PNG', 'JPEG', 'WEBP'}:
                            raise ValueError('Unsupported image encoding')
                        total_pixels += image.width * image.height
                        if (image.width * image.height > MAX_IMAGE_PIXELS or total_pixels > MAX_TOTAL_PIXELS
                                or getattr(image, 'n_frames', 1) != 1):
                            raise ValueError('Image pixel limit exceeded or animated image')
                        # Match native HF image loading for camera JPEGs: honor
                        # EXIF orientation before handing pixels to the processor.
                        decoded = ImageOps.exif_transpose(image).convert('RGB')
                        # Aspect-preserving, downscale-only cap in displayed (EXIF-
                        # corrected) coordinates. Native model processing still follows.
                        if max_image_width is not None or max_image_height is not None:
                            decoded.thumbnail((min(max_image_width or decoded.width, decoded.width),
                                               min(max_image_height or decoded.height, decoded.height)),
                                              resample=Image.Resampling.LANCZOS)
                        images.append(decoded)
            except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
                raise ValueError('Invalid or oversized image') from exc
            blocks.append({'type': 'image'})
        item['content'] = blocks
        result.append(item)
    return result, images


# llama.cpp (libmtmd) adapter


@dataclass(frozen=True, eq=False)
class MediaChunk:
    """One mtmd image chunk: n_tokens embeddings occupying n_pos positions.

    M-RoPE models (Qwen-VL family) place an image's embeddings on a 2-D grid, so
    the image advances the position counter by n_pos < n_tokens; other models
    use one position per embedding. placeholder is the first of the n_tokens
    request-unique negative IDs that stand for this chunk in Branch.token_ids.
    pointer is owned by its RequestImages and valid while that object lives.
    """

    pointer: int
    n_tokens: int
    n_pos: int
    placeholder: int

    def placeholder_ids(self):
        return [self.placeholder - k for k in range(self.n_tokens)]


class RequestImages:
    """Request-local tokenized images and their encoded embeddings.

    items[i] lists image i's native layout, e.g. [begin-marker text tokens,
    MediaChunk, end-marker text tokens]. Each image is tokenized on its own, so
    llama.cpp never merges two adjacent same-size images into one video frame
    pair: every image stays its own span, as in the upstream server. Encoded
    projector outputs are cached per chunk, so an image is encoded once per
    request however many question branches decode it. Nothing is shared across
    requests; mtmd chunk memory is released when this object is collected.
    """

    def __init__(self, items, containers):
        self.items = items
        self.embeddings = {}
        self._finalizer = weakref.finalize(self, _free_chunk_lists, list(containers))

    @property
    def chunks(self):
        return [item for layout in self.items for item in layout if isinstance(item, MediaChunk)]

    def close(self):
        self.embeddings.clear()
        self._finalizer()


def _free_chunk_lists(containers):
    if not containers:
        return
    from llama_cpp import mtmd_cpp
    for container in containers:
        mtmd_cpp.mtmd_input_chunks_free(container)


_LOG = logging.getLogger(__name__)


def _mtmd_log(level, text, _user_data):
    # GGML_LOG_LEVEL_WARN = 3, GGML_LOG_LEVEL_ERROR = 4. Per-image INFO lines
    # ("decoding image batch ...") would otherwise flood the server log.
    if level in (3, 4) and text:
        _LOG.warning(text.decode('utf-8', errors='replace').rstrip())


_MTMD_LOG_CALLBACK = None


def split_media_text(text, marker, count):
    """Split rendered chat at media markers; the marker count must match images."""
    parts = text.split(marker)
    if len(parts) - 1 != count:
        raise ValueError('Image placeholders must match supplied images; do not insert native media markers in text')
    return parts


class MtmdVision:
    """Own one mtmd context: a GGUF vision projector bound to the text model.

    Rendering follows llama-server's convention: each image block becomes the
    mtmd media marker as chat text, and mtmd wraps it with the model's own image
    begin/end tokens (e.g. Qwen's <|vision_start|>/<|vision_end|>, Gemma's
    <start_of_image>/<end_of_image>). Text between markers is tokenized
    independently with the GGUF vocabulary, exactly as mtmd_tokenize does.
    """

    def __init__(self, mmproj_path, model, *, use_gpu=False, n_threads=None):
        from llama_cpp import llama_cpp as lib, mtmd_cpp

        global _MTMD_LOG_CALLBACK
        if _MTMD_LOG_CALLBACK is None:
            _MTMD_LOG_CALLBACK = lib.llama_log_callback(_mtmd_log)
            mtmd_cpp.mtmd_log_set(_MTMD_LOG_CALLBACK, None)
            mtmd_cpp.mtmd_helper_log_set(_MTMD_LOG_CALLBACK, None)
        params = mtmd_cpp.mtmd_context_params_default()
        params.use_gpu = bool(use_gpu)
        params.print_timings = False
        if n_threads is not None:
            params.n_threads = int(n_threads)
        self.marker = mtmd_cpp.mtmd_default_marker().decode('utf-8')
        params.media_marker = self.marker.encode('utf-8')
        ctx = mtmd_cpp.mtmd_init_from_file(str(mmproj_path).encode('utf-8'), model, params)
        if not ctx:
            raise ValueError(f'Unable to load vision projector {mmproj_path!r} for this GGUF model')
        self.ctx = ctx
        self._finalizer = weakref.finalize(self, mtmd_cpp.mtmd_free, ctx)
        if not mtmd_cpp.mtmd_support_vision(ctx):
            self.close()
            raise ValueError(f'Projector {mmproj_path!r} does not support image input')
        self.mrope = bool(mtmd_cpp.mtmd_decode_use_mrope(ctx))
        self.n_embd = int(lib.llama_model_n_embd_inp(model))
        self.path = str(mmproj_path)

    def close(self):
        self._finalizer()

    def prepare(self, images):
        """Tokenize decoded RGB PIL images, each on its own; no model execution.

        mtmd_tokenize only preprocesses (resize/normalize to the projector's
        grid) and is thread-safe on a shared context, so compilation may run
        outside the backend lock. Encoding happens later, inside the backend.
        """
        from llama_cpp import mtmd_cpp

        items, containers, placeholder = [], [], -1
        try:
            for image in images:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                data = image.tobytes()
                buffer = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
                bitmap = mtmd_cpp.mtmd_bitmap_init(image.width, image.height, buffer)
                if not bitmap:
                    raise ValueError('Unable to prepare image for the vision projector')
                container = mtmd_cpp.mtmd_input_chunks_init()
                try:
                    if not container:
                        raise ValueError('Unable to allocate image chunks')
                    containers.append(container)
                    text = mtmd_cpp.mtmd_input_text()
                    encoded = self.marker.encode('utf-8')
                    text.text, text.text_len = encoded, len(encoded)
                    text.add_special, text.parse_special = False, True
                    bitmaps = (mtmd_cpp.mtmd_bitmap_p_ctypes * 1)(bitmap)
                    status = mtmd_cpp.mtmd_tokenize(self.ctx, container, ctypes.byref(text), bitmaps, 1)
                finally:
                    mtmd_cpp.mtmd_bitmap_free(bitmap)
                if status != 0:
                    raise ValueError('Vision projector could not preprocess the image')
                layout = []
                for index in range(mtmd_cpp.mtmd_input_chunks_size(container)):
                    chunk = mtmd_cpp.mtmd_input_chunks_get(container, index)
                    kind = mtmd_cpp.mtmd_input_chunk_get_type(chunk)
                    if kind == mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_TEXT:
                        count = ctypes.c_size_t()
                        tokens = mtmd_cpp.mtmd_input_chunk_get_tokens_text(chunk, ctypes.byref(count))
                        layout.append([int(tokens[k]) for k in range(count.value)])
                    elif kind == mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_IMAGE:
                        n_tokens = int(mtmd_cpp.mtmd_input_chunk_get_n_tokens(chunk))
                        n_pos = int(mtmd_cpp.mtmd_input_chunk_get_n_pos(chunk))
                        if n_tokens < 1 or not 1 <= n_pos <= n_tokens:
                            raise ValueError('Vision projector produced an invalid image span')
                        media = MediaChunk(int(chunk), n_tokens, n_pos, placeholder)
                        placeholder -= n_tokens
                        layout.append(media)
                    else:
                        raise ValueError('Only image media is supported')
                if not any(isinstance(item, MediaChunk) for item in layout):
                    raise ValueError('Vision projector produced no image tokens')
                items.append(layout)
        except BaseException:
            _free_chunk_lists(containers)
            raise
        return RequestImages(items, containers)

    def tokenize(self, tokenizer, text, request_images):
        """Assemble token IDs and media spans for one rendered branch prompt.

        Returns (ids, spans, tail): image embeddings appear in ids as their
        request-unique negative placeholder IDs, spans lists (start, chunk)
        pairs, and tail is the text after the final marker, whose tokens end ids.
        """
        parts = split_media_text(text, self.marker, len(request_images.items))
        ids = list(tokenizer.encode(parts[0], add_special_tokens=False))
        spans = []
        for layout, part in zip(request_images.items, parts[1:]):
            for item in layout:
                if isinstance(item, MediaChunk):
                    spans.append((len(ids), item))
                    ids.extend(item.placeholder_ids())
                else:
                    ids.extend(item)
            ids.extend(tokenizer.encode(part, add_special_tokens=False))
        return ids, tuple(spans), parts[-1]

    def encode(self, request_images, chunk):
        """Run the projector once per chunk and keep a private copy of its output."""
        from llama_cpp import mtmd_cpp

        cached = request_images.embeddings.get(chunk)
        if cached is not None:
            return cached
        if mtmd_cpp.mtmd_encode_chunk(self.ctx, chunk.pointer) != 0:
            raise RuntimeError('Vision projector failed to encode an image')
        output = mtmd_cpp.mtmd_get_output_embd(self.ctx)
        if not output:
            raise RuntimeError('Vision projector returned no embeddings')
        size = self.n_embd * chunk.n_tokens
        embeddings = (ctypes.c_float * size)()
        ctypes.memmove(embeddings, output, size * ctypes.sizeof(ctypes.c_float))
        request_images.embeddings[chunk] = embeddings
        return embeddings

    def decode(self, context, request_images, chunk, n_past, seq_id, n_batch):
        """Decode one image's embeddings into seq_id at position n_past.

        mtmd's helper sets the native M-RoPE 2-D positions and, where a model
        requires it (e.g. Gemma 3), non-causal attention inside the image span.
        Returns the next free position (n_past + chunk.n_pos).
        """
        from llama_cpp import llama_cpp as lib, mtmd_cpp

        embeddings = self.encode(request_images, chunk)
        new_n_past = lib.llama_pos(0)
        status = mtmd_cpp.mtmd_helper_decode_image_chunk(
            self.ctx, context, chunk.pointer, embeddings, int(n_past), int(seq_id),
            int(n_batch), ctypes.byref(new_n_past),
            mtmd_cpp.mtmd_helper_post_decode_callback(), None)  # NULL: no callback
        if status != 0:
            raise RuntimeError('llama_decode failed for an image span')
        if new_n_past.value != n_past + chunk.n_pos:
            raise RuntimeError('Unexpected image position advance')
        return new_n_past.value
