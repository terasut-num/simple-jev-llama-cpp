"""Train/evaluate RFDT answer-token distributions with Transformers Trainer.

Run normally for one device, or with torchrun for one process per GPU. Each
question is a training row, retaining its original request's full question
briefing. Only the final answer position and allowed vocabulary labels incur
loss. The entire context remains differentiable; inference caches are disabled.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from data import context_key, read_jsonl, target_distribution, validate_record
from hf_server import PromptCompiler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def load_model(name, dtype, revision=None):
    """Mirror the HF server's model-family selection, without inference device_map.

    Trainer/Accelerate places a model on each rank's device. device_map='auto'
    is deliberately absent: inference placement is not distributed training.
    """
    config = AutoConfig.from_pretrained(name, revision=revision)
    loader = (
        AutoModelForImageTextToText
        if config.model_type in {"gemma4", "qwen3_5", "qwen3_5_moe"}
        else AutoModelForCausalLM
    )
    return loader.from_pretrained(name, revision=revision, dtype=getattr(torch, dtype))


def compile_rows(records, tokenizer, max_length):
    """Reject missing targets and long/unstable prompts instead of truncating them."""
    rows = []
    for record in records:
        request, plan = validate_record(record)
        if set(record.get("targets", {})) != set(request.questions):
            raise ValueError("Run prepare.py first: every question needs a target")
        compiled = PromptCompiler(
            tokenizer, max_tokens=max_length, version=plan.template_version
        ).compile(request)
        for question, branch in zip(plan.questions, compiled.branches):
            target = target_distribution(
                request.questions[question.question_id],
                record["targets"][question.question_id],
            )
            rows.append(
                {
                    "input_ids": branch.token_ids,
                    "allowed_ids": branch.output_ids,
                    "labels": target,
                }
            )
    return rows


class DecisionCollator:
    """Right-pad prompts and label sets separately; padding has no loss contribution."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        batch, width, choices = (
            len(rows),
            max(len(r["input_ids"]) for r in rows),
            max(len(r["allowed_ids"]) for r in rows),
        )
        ids = torch.full((batch, width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        allowed = torch.zeros((batch, choices), dtype=torch.long)
        labels = torch.zeros((batch, choices), dtype=torch.float32)
        allowed_mask = torch.zeros((batch, choices), dtype=torch.bool)
        for i, row in enumerate(rows):
            size, count = len(row["input_ids"]), len(row["allowed_ids"])
            ids[i, :size] = torch.tensor(row["input_ids"])
            mask[i, :size] = 1
            allowed[i, :count] = torch.tensor(row["allowed_ids"])
            labels[i, :count] = torch.tensor(row["labels"])
            allowed_mask[i, :count] = True
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "allowed_ids": allowed,
            "allowed_mask": allowed_mask,
            "labels": labels,
        }


def decision_logits(model, inputs):
    """Gather the real final position BEFORE restricting to each row's labels.

    This portable first implementation materializes vocabulary logits for all
    positions. Small batches and gradient checkpointing are recommended. A later
    optimization can project only answer positions without changing this loss.
    """
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        use_cache=False,
    )
    positions = inputs["attention_mask"].sum(dim=1) - 1
    final = outputs.logits[
        torch.arange(len(positions), device=positions.device), positions
    ]
    selected = final.gather(1, inputs["allowed_ids"]).float()
    # A finite sentinel avoids 0 * -inf in soft-label cross entropy.
    return selected.masked_fill(~inputs["allowed_mask"], -1e9)


class DecisionTrainer(Trainer):
    """Soft cross-entropy equals teacher→student KL up to target entropy.

    One-hot targets reduce to ordinary classification cross-entropy. Never pass
    the target distributions as language-model labels: that would train the wrong
    positions/vocabulary. Trainer owns backward, accumulation, AMP, and DDP.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        selected = decision_logits(model, inputs)
        loss = -(inputs["labels"] * selected.log_softmax(dim=-1)).sum(dim=-1).mean()
        return (loss, {"logits": selected}) if return_outputs else loss


def metrics(prediction):
    """Compare target distributions; mode agreement isn't accuracy for soft labels.

    Trainer pads variable-width predictions/labels with -100 across batches.
    Our collator uses zero-probability padding within a batch. Both are removed
    before normalization. No score is claimed to measure probability calibration.
    """
    logits, targets = prediction
    valid = (targets >= 0) & (logits > -1e8)
    logits = np.where(valid, logits, -1e9)
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    targets = np.maximum(targets, 0)
    kl = np.sum(
        np.where(
            targets > 0,
            targets
            * (
                np.log(np.maximum(targets, 1e-30))
                - np.log(np.maximum(probabilities, 1e-30))
            ),
            0,
        ),
        axis=1,
    )
    return {
        "target_mode_agreement": float(
            (probabilities.argmax(1) == targets.argmax(1)).mean()
        ),
        "mean_target_kl": float(kl.mean()),
        "mean_distribution_l1": float(np.abs(probabilities - targets).sum(1).mean()),
    }


def assert_disjoint(train, validation):
    """Catch leakage even if callers supplied split files manually."""

    def keys(rows):
        result = set()
        for row in rows:
            request, _ = validate_record(row)
            result.add(("context", context_key(request)))
            if row.get("group_id"):
                result.add(("source", row["group_id"]))
        return result

    if keys(train) & keys(validation):
        raise ValueError("Training and validation share a context or group_id")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--train")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--adapter", help="Existing LoRA adapter to evaluate (requires --eval-only)"
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Train an adapter instead of all model weights",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="float32"
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", help="Trainer checkpoint directory")
    args = parser.parse_args()
    if not args.eval_only and not args.train:
        parser.error("--train is required unless --eval-only")
    if (args.adapter and not args.eval_only) or (args.lora and args.eval_only):
        parser.error("Use --lora to train; --adapter with --eval-only to evaluate")
    if (
        min(
            args.batch_size, args.gradient_accumulation, args.max_length, args.lora_rank
        )
        < 1
    ):
        parser.error(
            "batch size, accumulation, max length, and LoRA rank must be positive"
        )
    # Avoid overwriting another run's weights. Resume checkpoints can share output.
    output = Path(args.output)
    if (
        not args.eval_only
        and not args.resume
        and output.exists()
        and any(output.iterdir())
    ):
        parser.error("Output directory is nonempty; choose a new directory or --resume")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Tokenizer needs a pad or EOS token")
    validation_records = read_jsonl(args.validation)
    training_records = read_jsonl(args.train) if not args.eval_only else []
    assert_disjoint(training_records, validation_records)
    validation_rows = compile_rows(validation_records, tokenizer, args.max_length)
    training_rows = compile_rows(training_records, tokenizer, args.max_length)
    model = load_model(args.model, args.dtype, args.revision)
    if args.lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=2 * args.lora_rank,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
        )
    elif args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    configuration = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        data_seed=args.seed,
        use_cpu=args.cpu,
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        label_names=["labels"],
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        logging_first_step=True,
        report_to="none",
        optim="adamw_torch",
        ddp_find_unused_parameters=True,
    )
    trainer = DecisionTrainer(
        model=model,
        args=configuration,
        train_dataset=training_rows or None,
        eval_dataset=validation_rows,
        data_collator=DecisionCollator(pad_id),
        compute_metrics=metrics,
    )
    if not args.eval_only:
        trainer.train(resume_from_checkpoint=args.resume)
        trainer.save_model(args.output)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(args.output)
            (output / "rfdt_config.json").write_text(
                json.dumps(
                    {
                        **vars(args),
                        "template_versions": sorted(
                            {r.get("template_version", "v1") for r in training_records}
                        ),
                        "loss": "allowed_label_soft_cross_entropy",
                    },
                    indent=2,
                )
            , encoding='utf-8', newline='\n')
    results = trainer.evaluate()
    trainer.log_metrics("eval", results)
    trainer.save_metrics("eval", results)


if __name__ == "__main__":
    main()
