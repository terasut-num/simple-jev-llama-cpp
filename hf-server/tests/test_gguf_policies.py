"""GGUF counterparts of the HF prompt-policy plumbing, without model weights.

Covers the GGUF header reader, the header-to-profile fingerprint that drives
automatic prompt-policy selection, LlamaCppTokenizer's continue_final_message
and special-token surface, the startup probe, and --chat-template-file.
Header dictionaries follow what llama.cpp's convert_hf_to_gguf.py writes.
"""

import importlib.util
import struct
import sys
from unittest.mock import Mock

import pytest

from conftest import QWEN35_4B_GGUF, Tokenizer, fake_gguf_loader
from hf_prompt_policies import KNOWN_PROFILES, resolve_prompt_policy
from hf_server import (
    CONTINUE_FINAL_MESSAGE_TAG,
    LlamaCppTokenizer,
    gguf_backbone_config,
    load_service,
    read_gguf_metadata,
)
from test_prompt_policies import NativeTokenizer


def write_gguf(path, values):
    """Write a tensor-free GGUF v3 header holding values' types as llama.cpp does."""

    def string(text):
        data = text.encode()
        return struct.pack("<Q", len(data)) + data

    def typed(value):
        if isinstance(value, bool):
            return 7, struct.pack("<?", value)
        if isinstance(value, int):
            return 4, struct.pack("<I", value)
        if isinstance(value, float):
            return 6, struct.pack("<f", value)
        if isinstance(value, str):
            return 8, string(value)
        kind = typed(value[0])[0]
        body = b"".join(
            string(v) if kind == 8 else typed(v)[1] for v in value
        )
        return 9, struct.pack("<IQ", kind, len(value)) + body

    blob = b"GGUF" + struct.pack("<IQQ", 3, 0, len(values))
    for key, value in values.items():
        kind, data = typed(value)
        blob += string(key) + struct.pack("<I", kind) + data
    path.write_bytes(blob)


def test_header_reader_keeps_arrays_and_skips_strings(tmp_path):
    path = tmp_path / "header.gguf"
    write_gguf(path, {
        "general.architecture": "gemma4",
        "gemma4.block_count": 6,
        "gemma4.attention.head_count_kv": [8, 8, 8, 8, 8, 2],
        "gemma4.attention.sliding_window_pattern": [True] * 5 + [False],
        "gemma4.rope.freq_base": 10000.0,
        "tokenizer.ggml.tokens": ["<pad>", "a", "b"],
        "tokenizer.chat_template": "{{ messages }}",
    })
    values = read_gguf_metadata(path)
    assert values == {
        "general.architecture": "gemma4",
        "gemma4.block_count": 6,
        "gemma4.attention.head_count_kv": [8, 8, 8, 8, 8, 2],
        "gemma4.attention.sliding_window_pattern": [True] * 5 + [False],
        "gemma4.rope.freq_base": 10000.0,
        "tokenizer.chat_template": "{{ messages }}",
    }
    (tmp_path / "bad.gguf").write_bytes(b"GGML" + bytes(20))
    with pytest.raises(ValueError, match="GGUF"):
        read_gguf_metadata(tmp_path / "bad.gguf")


GEMMA_SWA = [True] * 5 + [False]
# (profile name, GGUF header, vocabulary size) per KNOWN_PROFILES entry.
GGUF_PROFILES = [
    ("Qwen dense 4B", QWEN35_4B_GGUF, 248320),
    ("Qwen dense 27B", {
        "general.architecture": "qwen35", "qwen35.block_count": 64,
        "qwen35.embedding_length": 5120, "qwen35.attention.head_count": 24,
        "qwen35.attention.head_count_kv": 4, "qwen35.attention.key_length": 256,
        "qwen35.feed_forward_length": 17408,
    }, 248320),
    # MTP export appends nextn layers to block_count.
    ("Qwen MoE 35B-A3B", {
        "general.architecture": "qwen35moe", "qwen35moe.block_count": 41,
        "qwen35moe.nextn_predict_layers": 1,
        "qwen35moe.embedding_length": 2048, "qwen35moe.attention.head_count": 16,
        "qwen35moe.attention.head_count_kv": 2, "qwen35moe.attention.key_length": 256,
        "qwen35moe.expert_count": 256, "qwen35moe.expert_feed_forward_length": 512,
        "qwen35moe.expert_used_count": 8,
        "qwen35moe.expert_shared_feed_forward_length": 512,
    }, 248320),
    ("Gemma unified dense 12B", {
        "general.architecture": "gemma4", "gemma4.block_count": 48,
        "gemma4.embedding_length": 3840, "gemma4.attention.head_count": 16,
        "gemma4.attention.head_count_kv": [8, 8, 8, 8, 8, 4] * 8,
        "gemma4.attention.sliding_window_pattern": GEMMA_SWA * 8,
        "gemma4.attention.key_length": 512, "gemma4.attention.key_length_swa": 256,
        "gemma4.feed_forward_length": 15360,
    }, 262144),
    # Values read from the header of llama.cpp's Gemma 4 26B-A4B vocab GGUF.
    ("Gemma MoE 26B-A4B", {
        "general.architecture": "gemma4", "gemma4.block_count": 30,
        "gemma4.embedding_length": 2816, "gemma4.attention.head_count": 16,
        "gemma4.attention.head_count_kv": [8, 8, 8, 8, 8, 2] * 5,
        "gemma4.attention.sliding_window_pattern": GEMMA_SWA * 5,
        "gemma4.attention.key_length": 512, "gemma4.attention.key_length_swa": 256,
        "gemma4.feed_forward_length": 2112, "gemma4.expert_count": 128,
        "gemma4.expert_feed_forward_length": 704, "gemma4.expert_used_count": 8,
    }, 262144),
]


@pytest.mark.parametrize("name,header,vocab", GGUF_PROFILES)
def test_gguf_headers_select_known_profiles(name, header, vocab, tmp_path):
    expected = {profile: policy for profile, policy, _ in KNOWN_PROFILES}[name]
    path = tmp_path / "renamed-anything.gguf"
    write_gguf(path, header)
    config = gguf_backbone_config(read_gguf_metadata(path), vocab)
    policy, selection = resolve_prompt_policy(config)
    assert (policy, selection["profile"]) == (expected, name)
    # A nearby size is not assigned a guessed policy.
    wider = dict(header)
    key = f"{header['general.architecture']}.embedding_length"
    wider[key] += 1
    assert resolve_prompt_policy(gguf_backbone_config(wider, vocab))[0] == "baseline"


def test_unprofiled_gguf_architectures_warn_and_use_baseline(caplog):
    llama = {
        "general.architecture": "llama", "llama.block_count": 32,
        "llama.embedding_length": 2560, "llama.attention.head_count": 16,
        "llama.attention.head_count_kv": 4, "llama.feed_forward_length": 9216,
    }
    config = gguf_backbone_config(llama, 248320)
    assert config["model_type"] == "gguf:llama"
    policy, selection = resolve_prompt_policy(config)
    assert (policy, selection["mode"]) == ("baseline", "unknown-baseline")
    assert "UNRECOGNIZED MODEL ARCHITECTURE/SIZE" in caplog.text
    # The same Qwen dimensions under a different vocabulary are unrecognized.
    assert resolve_prompt_policy(gguf_backbone_config(QWEN35_4B_GGUF, 151936))[0] == "baseline"


class FakeVocab:
    """Metadata/token surface of a vocabulary-only llama.cpp model."""

    def __init__(self, template):
        self.model = "vocab"
        self._metadata = {
            "tokenizer.chat_template": template,
            "tokenizer.ggml.bos_token_id": "1",
            "tokenizer.ggml.eos_token_id": "2",
        }

    def metadata(self):
        return self._metadata

    def token_get_text(self, token):
        return {1: "<s>", 2: "</s>"}[token]


# Close each turn with an end marker; optionally trim content like many models.
TEMPLATE = (
    "{% for m in messages %}<|{{ m.role }}|>"
    "{% if m.reasoning_content is defined and enable_thinking %}<think>{{ m.reasoning_content }}</think>{% endif %}"
    "{% if m.content is string %}{{ m.content }}{% else %}{% for b in m.content %}{{ b.text }}{% endfor %}{% endif %}"
    "<|end|>\n{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
TRIMMING = TEMPLATE.replace("{{ b.text }}", "{{ b.text | trim }}")


def blocks(messages):
    return [{**m, "content": [{"type": "text", "text": m["content"]}]} for m in messages]


MESSAGES = [
    {"role": "user", "content": "Is it a cat?"},
    {"role": "assistant", "content": '{"answer": ', "reasoning_content": "[thinking]\n"},
]


def test_continue_final_message_opens_the_assistant_turn():
    tokenizer = LlamaCppTokenizer(FakeVocab(TEMPLATE))
    text = tokenizer.apply_chat_template(
        blocks(MESSAGES), tokenize=False, add_generation_prompt=False,
        continue_final_message=True, enable_thinking=True,
    )
    # Trailing spacing survives; the assistant end marker is cut off.
    assert text == '<|user|>Is it a cat?<|end|>\n<|assistant|><think>[thinking]\n</think>{"answer": '
    assert CONTINUE_FINAL_MESSAGE_TAG.strip() not in text
    # String content is continued the same way; thinking follows the flag.
    text = tokenizer.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=False,
        continue_final_message=True, enable_thinking=False,
    )
    assert text.endswith('<|assistant|>{"answer": ')
    # Templates that trim the message lose the trailing space, like HF.
    trimmed = LlamaCppTokenizer(FakeVocab(TRIMMING)).apply_chat_template(
        blocks(MESSAGES), tokenize=False, add_generation_prompt=False,
        continue_final_message=True, enable_thinking=True,
    )
    assert trimmed.endswith('</think>{"answer":')
    # The caller's messages are not mutated by the continuation tag.
    assert MESSAGES[-1]["content"] == '{"answer": '


def test_continue_final_message_errors():
    tokenizer = LlamaCppTokenizer(FakeVocab(TEMPLATE))
    with pytest.raises(ValueError, match="not compatible"):
        tokenizer.apply_chat_template(
            MESSAGES, tokenize=False, add_generation_prompt=True, continue_final_message=True,
        )
    dropping = LlamaCppTokenizer(FakeVocab(
        "{% for m in messages %}{% if m.role == 'user' %}{{ m.content }}{% endif %}{% endfor %}"
    ))
    with pytest.raises(ValueError, match="does not appear"):
        dropping.apply_chat_template(
            MESSAGES, tokenize=False, add_generation_prompt=False, continue_final_message=True,
        )
    with pytest.raises(ValueError, match="text only"):
        tokenizer.apply_chat_template(MESSAGES, tokenize=True)
    with pytest.raises(ValueError, match="chat_template"):
        LlamaCppTokenizer(FakeVocab(""))


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None,
                    reason="Transformers parity check needs transformers")
@pytest.mark.parametrize("template", [TEMPLATE, TRIMMING])
@pytest.mark.parametrize("wrap", [blocks, list])
def test_continue_final_message_matches_transformers(template, wrap):
    from transformers.utils.chat_template_utils import render_jinja_template

    tokenizer = LlamaCppTokenizer(FakeVocab(template))
    for thinking in (True, False):
        messages = wrap(MESSAGES)
        ours = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            continue_final_message=True, enable_thinking=thinking,
        )
        reference = render_jinja_template(
            [messages], chat_template=template, bos_token="<s>", eos_token="</s>",
            add_generation_prompt=False, continue_final_message=True,
            enable_thinking=thinking,
        )[0][0]
        assert ours == reference


def test_special_ids_are_control_tokens(monkeypatch):
    import llama_cpp

    attributes = {0: 0, 1: llama_cpp.LLAMA_TOKEN_ATTR_CONTROL, 2: 0,
                  3: llama_cpp.LLAMA_TOKEN_ATTR_CONTROL | 1}
    calls = []
    monkeypatch.setattr(llama_cpp, "llama_model_get_vocab", lambda model: "vocab")
    monkeypatch.setattr(llama_cpp, "llama_vocab_n_tokens", lambda vocab: len(attributes))
    monkeypatch.setattr(llama_cpp, "llama_vocab_get_attr",
                        lambda vocab, token: calls.append(token) or attributes[token])
    tokenizer = LlamaCppTokenizer(FakeVocab(TEMPLATE))
    assert tokenizer.all_special_ids == {1, 3}
    assert tokenizer.all_special_ids == {1, 3}
    assert calls == [0, 1, 2, 3]  # Scanned once.


class NoReasoning(NativeTokenizer):
    """Native renderer whose template discards assistant reasoning_content."""

    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get("add_generation_prompt"):
            return super().apply_chat_template(messages, **kwargs)
        return "\n".join(f"{m['role']}: {m['content'][0]['text']}" for m in messages)


# None auto-selects the Qwen dense 4B profile's shared_examples_binary format.
@pytest.mark.parametrize("requested,policy", [(None, "shared_examples_binary"),
                                              ("strict_mix_repeat2", "strict_mix_repeat2")])
def test_startup_probe_rejects_template_that_cannot_serve_policy(monkeypatch, requested, policy):
    created = fake_gguf_loader(monkeypatch, NoReasoning(), QWEN35_4B_GGUF)
    with pytest.raises(ValueError, match="--chat-template-file") as error:
        load_service("model.gguf", prompt_policy=requested, max_choice_options=50)
    assert policy in str(error.value)
    assert "did not preserve" in str(error.value)
    assert [m.vocab_only for m in created.models] == [True]  # No weights loaded.
    service = load_service("model.gguf", prompt_policy="baseline", max_choice_options=50)
    assert service.compiler.prompt_policy == "baseline"


def test_chat_template_file_overrides_embedded_template(monkeypatch, tmp_path):
    template = tmp_path / "newer.jinja"
    template.write_text("{{ messages }}", encoding="utf-8")
    created = fake_gguf_loader(monkeypatch, NativeTokenizer(), QWEN35_4B_GGUF)
    service = load_service("model.gguf", chat_template_file=str(template), max_choice_options=50)
    # Both the startup probe and the served compiler use the replacement.
    assert [t for _, t in created.tokenizers] == ["{{ messages }}", "{{ messages }}"]
    assert service.metadata["chat_template_source"] == str(template)
    default = load_service("model.gguf", max_choice_options=50)
    assert default.metadata["chat_template_source"] == "gguf"


def test_cli_gguf_flags(monkeypatch):
    import uvicorn

    import hf_server

    load = Mock()
    monkeypatch.setattr(hf_server, "load_service", load)
    monkeypatch.setattr(hf_server, "create_app", lambda service: service)
    monkeypatch.setattr(uvicorn, "run", Mock())
    monkeypatch.setattr(sys, "argv", [
        "simple-jev", "--model", "org/model-GGUF", "--gguf-file", "model-Q4_K_M.gguf",
        "--chat-template-file", "chat.jinja", "--classifier-prompt-policy", "repeat_state",
        "--served-model-name", "public", "--n-gpu-layers", "12", "--prefix-sharing", "off",
    ])
    hf_server.main()
    assert load.call_args.args == ("org/model-GGUF",)
    kwargs = load.call_args.kwargs
    assert kwargs["gguf_file"] == "model-Q4_K_M.gguf"
    assert kwargs["chat_template_file"] == "chat.jinja"
    assert kwargs["prompt_policy"] == "repeat_state"
    assert kwargs["served_model_name"] == "public"
    assert kwargs["n_gpu_layers"] == 12
    assert kwargs["prefix_sharing"] == "off"


def test_invalid_loader_options_fail_before_any_file_access(monkeypatch):
    created = fake_gguf_loader(monkeypatch, Tokenizer())
    with pytest.raises(ValueError, match="prefix sharing"):
        load_service("model.gguf", prefix_sharing="sometimes")
    with pytest.raises(ValueError, match="Unknown device"):
        load_service("model.gguf", device="tpu")
    with pytest.raises(ValueError, match="served_model_name"):
        load_service("model.gguf", served_model_name=" ")
    assert created.paths == [] and created.models == []
