"""RFDT's on-disk data contract and conversion from answers to distributions.

Requests keep their complete question set, even when training one question at a
time: v1 briefs the model on all questions before presenting the context.
Targets use public candidate IDs, zero-based rubric indices, or Noul bins 1–9.
They never contain tokenizer-specific token IDs, so teachers and students can
use different tokenizers. This module needs only the existing HF installation.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hf-server"))

from common import ClassifierRequest, prepare_prompt


def read_jsonl(path):
    """Load records with actionable line numbers; never silently skip bad data."""
    rows = []
    with open(path, encoding='utf-8') as stream:
        for line, text in enumerate(stream, 1):
            if not text.strip():
                continue
            try:
                row = json.loads(text)
                if not isinstance(row, dict):
                    raise TypeError("expected a JSON object")
                rows.append(row)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line}: {error}") from error
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def fingerprint(value):
    """Preserve insertion order: candidate and question order affect the prompt."""
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def context_key(request):
    """Group identical contexts even when questions/model IDs differ."""
    value = {
        "state": request.state,
        "messages": (
            [m.model_dump(exclude_none=True) for m in request.messages]
            if request.messages is not None
            else None
        ),
    }
    # Context object key order has no semantic significance in the HF renderer.
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def validate_record(row):
    """Return a request/plan; training metadata never goes into ClassifierRequest."""
    extra = set(row) - {
        "id",
        "group_id",
        "template_version",
        "request",
        "targets",
        "provenance",
    }
    if extra:
        raise ValueError(f"Unknown record fields: {sorted(extra)}")
    request = ClassifierRequest.model_validate(row["request"])
    plan = prepare_prompt(request, version=row.get("template_version", "v1"))
    targets = row.get("targets", {})
    if not isinstance(targets, dict) or set(targets) - set(request.questions):
        raise ValueError("targets must be an object keyed by known question IDs")
    if "group_id" in row and (
        not isinstance(row["group_id"], str) or not row["group_id"]
    ):
        raise ValueError("group_id must be a nonempty string")
    return request, plan


def target_distribution(question, target):
    """Convert one answer or an explicit distribution to ordered probabilities.

    Fractional rubric scores interpolate adjacent levels. Noul scalars invert
    v1's public 0.01–0.99 mapping then interpolate adjacent rating bins. This is
    a chosen supervision rule, NOT reconstruction of a teacher distribution.
    Explicit probabilities must cover every public label and sum to one.
    """
    if not isinstance(target, dict) or set(target) not in (
        {"answer"},
        {"probabilities"},
    ):
        raise ValueError("Each target needs exactly answer OR probabilities")
    kind = question.type
    labels = (
        list(question.criteria)
        if kind == "choice"
        else (
            [str(i) for i in range(len(question.criteria))]
            if kind == "score"
            else list("123456789")
        )
    )
    if "probabilities" in target:
        values = target["probabilities"]
        if not isinstance(values, dict) or set(values) != set(labels):
            raise ValueError(f"probabilities must contain exactly {labels}")
        probabilities = [values[label] for label in labels]
        if any(
            isinstance(p, bool)
            or not isinstance(p, (float, int))
            or not math.isfinite(p)
            or p < 0
            for p in probabilities
        ):
            raise ValueError("probabilities must be finite nonnegative numbers")
        total = sum(probabilities)
        if not math.isclose(total, 1.0, abs_tol=1e-5):
            raise ValueError("probabilities must sum to one")
        return [p / total for p in probabilities]
    answer = target["answer"]
    if kind == "choice":
        if not isinstance(answer, str) or answer not in labels:
            raise ValueError(f"choice answer must be one of {labels}")
        return [float(label == answer) for label in labels]
    if isinstance(answer, bool):
        if kind != "noul":
            raise ValueError("score answer must be numeric")
        answer = 0.99 if answer else 0.01
    if not isinstance(answer, (int, float)) or not math.isfinite(answer):
        raise ValueError("score/noul answer must be finite and numeric")
    if kind == "noul":
        if not 0.01 <= answer <= 0.99:
            raise ValueError("noul answer must be in [0.01, 0.99], or a boolean")
        position = (answer - 0.01) * 8 / 0.98
    else:
        position = answer
    if not 0 <= position <= len(labels) - 1:
        raise ValueError("score answer outside rubric range")
    low = math.floor(position)
    high = min(low + 1, len(labels) - 1)
    result = [0.0] * len(labels)
    result[low] += 1 - (position - low)
    result[high] += position - low
    return result


def normalized_targets(request, targets):
    """Store distributions in public-label order, independent of teacher tokens."""
    result = {}
    for key, target in targets.items():
        question = request.questions[key]
        labels = (
            list(question.criteria)
            if question.type == "choice"
            else (
                [str(i) for i in range(len(question.criteria))]
                if question.type == "score"
                else list("123456789")
            )
        )
        result[key] = {
            "probabilities": dict(zip(labels, target_distribution(question, target)))
        }
    return result
