"""Real LoRA fine-tune callables (HuggingFace + PEFT) — a TEMPLATE.

This is the reference implementation behind the mlprobe LoRA example. It is
*not* exercised by the dep-free `run.py` paths; importing it requires the heavy
deps and running a real probe needs a GPU:

    pip install -e ".[demo]" peft

Adapt the tokenization / prompt formatting to your dataset, and point
``_HF_IDS`` at models you can access (the defaults are gated/large — for a CPU
smoke test, map a base_model to a tiny model like
``hf-internal-testing/tiny-random-LlamaForCausalLM``).

The only mlprobe-specific contract is the three callables below: honor the
``runtime`` knobs, return ``TrainResult.artifact_path`` (here the LoRA adapter)
and metrics (include ``input_rows`` so the subset-fraction guard works), and
return quality metrics keyed by the names declared in the spec.
"""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from mlprobe import EvalResult, RuntimeConfig, TrainResult


# Short names used in the spec -> actual HF model ids you can access.
_HF_IDS = {
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
}
_DATASET = "tatsu-lab/alpaca"          # swap for your instruction data
_MAX_SEQ_LEN = 1024


def _dtype(precision: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, torch.float32)


def load(subset_fraction: float = 1.0, split: str | None = None, seed: int | None = None):
    ds = load_dataset(_DATASET, split=split or "train")
    if subset_fraction < 1.0:
        n = max(1, int(len(ds) * subset_fraction))
        ds = ds.shuffle(seed=seed if seed is not None else 0).select(range(n))
    return ds


def train(config, dataset_subset, output_dir, resume_from=None, runtime=RuntimeConfig()):
    output_dir = Path(output_dir)
    hf_id = _HF_IDS.get(config["base_model"], config["base_model"])

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=_dtype(runtime.precision))
    model = get_peft_model(model, LoraConfig(
        r=config["lora_r"], lora_alpha=config["lora_alpha"],
        target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=0.05, task_type="CAUSAL_LM"))

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=_MAX_SEQ_LEN)

    tokenized = dataset_subset.map(_tok, batched=True, remove_columns=dataset_subset.column_names)

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=runtime.gradient_accumulation_steps or 1,
        num_train_epochs=config["epochs"],
        bf16=(runtime.precision == "bf16"),
        fp16=(runtime.precision == "fp16"),
        gradient_checkpointing=True,
        seed=runtime.seed or 42,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    final_loss = next((h["loss"] for h in reversed(trainer.state.log_history) if "loss" in h), None)
    return TrainResult(
        artifact_path=adapter_dir,
        metrics={"train_loss": float(final_loss) if final_loss is not None else float("nan"),
                 "input_rows": float(len(dataset_subset))},
    )


def evaluate(artifact_path, eval_set) -> EvalResult:
    artifact_path = Path(artifact_path)
    tokenizer = AutoTokenizer.from_pretrained(artifact_path)
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(_base_of(artifact_path)), artifact_path)
    model.eval()

    # Replace with your task's real metric (exact-match / rougeL / a judge score).
    correct = total = 0
    for ex in eval_set:
        prompt, gold = ex["prompt"], ex["answer"]
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64)
        pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        correct += int(pred.strip() == gold.strip())
        total += 1
    return EvalResult(metrics={"exact_match": correct / max(total, 1)})


def _base_of(adapter_path: Path) -> str:
    """Read the base model id PEFT recorded in the adapter config."""
    import json

    cfg = json.loads((Path(adapter_path) / "adapter_config.json").read_text())
    return cfg["base_model_name_or_path"]
