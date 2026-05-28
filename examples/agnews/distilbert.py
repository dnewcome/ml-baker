"""DistilBERT fine-tune trainable for AG News.

The GPU end of the demo. ~94-95% test accuracy after a single epoch on the
full dataset, but training is dramatically slower than the sklearn baseline
— that's the cost/quality tradeoff the Pareto frontier exists to surface.

Honors the runner-injected ``runtime``:
  - ``runtime.precision`` → fp16 / bf16 / fp32 mixed-precision flag
  - ``runtime.n_gpus``    → no explicit handling; HF Trainer auto-detects
                            CUDA devices. When 0, runs on CPU (slow).
  - ``runtime.seed``      → torch + Trainer seed
  - ``runtime.n_cpus``    → DataLoader worker count
"""

from __future__ import annotations

from pathlib import Path

from mlprobe import EvalResult, RuntimeConfig, TrainResult


MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 4
MAX_LENGTH = 128


def train(config: dict, dataset_subset, output_dir: Path,
          resume_from: Path | None = None,
          runtime: RuntimeConfig = RuntimeConfig()) -> TrainResult:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    if runtime.seed is not None:
        torch.manual_seed(runtime.seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_source = str(resume_from) if resume_from else MODEL_NAME
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source, num_labels=NUM_LABELS,
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    tokenized = dataset_subset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")

    # Mixed precision only on GPU. Trainer silently ignores fp16/bf16 on CPU,
    # but being explicit prevents confusing warnings in the log.
    cuda_available = torch.cuda.is_available()
    fp16 = cuda_available and runtime.precision == "fp16"
    bf16 = cuda_available and runtime.precision == "bf16"

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=int(config.get("epochs", 1)),
        per_device_train_batch_size=int(config.get("batch_size", 16)),
        learning_rate=float(config.get("lr", 5e-5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        fp16=fp16,
        bf16=bf16,
        save_strategy="no",            # we save once at the end ourselves
        logging_steps=100,
        report_to=[],                  # don't try to init wandb/tensorboard
        seed=runtime.seed or 42,
        dataloader_num_workers=max(0, runtime.n_cpus - 1),
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorWithPadding(tokenizer),
        tokenizer=tokenizer,
    )
    train_out = trainer.train()

    artifact = output_dir / "model"
    trainer.save_model(str(artifact))
    tokenizer.save_pretrained(str(artifact))

    return TrainResult(
        artifact_path=artifact,
        metrics={"train_loss": float(train_out.training_loss)},
        steps_completed=int(train_out.global_step),
    )


def evaluate(artifact_path: Path, eval_set) -> EvalResult:
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(artifact_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(artifact_path))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    preds: list[int] = []
    labels: list[int] = []
    batch_size = 32
    texts_all = eval_set["text"]
    labels_all = eval_set["label"]

    for i in range(0, len(texts_all), batch_size):
        batch_texts = texts_all[i:i + batch_size]
        batch_labels = labels_all[i:i + batch_size]
        enc = tokenizer(
            batch_texts, truncation=True, max_length=MAX_LENGTH,
            padding=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
        labels.extend(int(x) for x in batch_labels)

    return EvalResult(metrics={
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    })
