"""QLoRA training and inference for COBOL test-suite generation."""

from data_quality.schema import CobolTask, ReviewStatus

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from data_quality.schema import CobolTask


def generation_prompt(task: CobolTask) -> str:
    return (
        "Generate a JSON test suite for this COBOL program. Use only observable "
        "behavior from the specification. Return JSON with a 'cases' array; each "
        "case has 'case_id', 'stdin', and 'expected_stdout'. Include boundary and "
        "error cases when specified. Do not include markdown.\n\n"
        f"Specification:\n{task.specification}\n\n"
        f"COBOL program:\n{task.program}"
    )


def _target(task: CobolTask) -> str:
    return json.dumps(
        {"cases": [case.__dict__ for case in task.reference_tests]},
        ensure_ascii=False,
        sort_keys=True,
    )


def _encode(tokenizer, task: CobolTask, max_length: int) -> dict[str, list[int]]:
    user = {"role": "user", "content": generation_prompt(task)}
    assistant = {"role": "assistant", "content": _target(task)}
    prefix = tokenizer.apply_chat_template(
        [user], tokenize=True, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        [user, assistant], tokenize=True, add_generation_prompt=False
    )
    if full[:len(prefix)] != prefix:
        raise ValueError(f"{task.task_id}: tokenizer prompt prefix mismatch")
    if len(prefix) >= max_length:
        raise ValueError(f"{task.task_id}: prompt exceeds max sequence length")
    input_ids = full[:max_length]
    labels = [-100] * min(len(prefix), len(input_ids))
    labels.extend(input_ids[len(labels):])
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class _Rows:
    def __init__(self, rows: list[dict[str, list[int]]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in self.rows[index].items()
        }


def _collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    import torch

    width = max(len(row["input_ids"]) for row in batch)
    ids, masks, labels = [], [], []
    for row in batch:
        padding = width - len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_id] * padding)
        masks.append(row["attention_mask"] + [0] * padding)
        labels.append(row["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def train_qlora(
    tasks: tuple[CobolTask, ...],
    output_dir: str | Path,
    model_id: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    seed: int = 2026,
    epochs: float = 2.0,
    learning_rate: float = 1e-4,
    max_length: int = 1024,
) -> Path:
    if not tasks:
        raise ValueError("train split is empty")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("duplicate training task IDs")

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("install Transformers, PEFT, and CUDA-compatible bitsandbytes") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a CUDA GPU")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization,
        device_map="auto",
    )
    resolved_revision = getattr(model.config, "_commit_hash", None) or "unknown"
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )

    dataset = _Rows([_encode(tokenizer, task, max_length) for task in tasks])
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(destination / "trainer"),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=seed,
        data_seed=seed,
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=lambda batch: _collate(batch, tokenizer.pad_token_id),
    )
    started = time.perf_counter()
    trained = trainer.train()
    elapsed = time.perf_counter() - started
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(str(destination))

    fingerprint = hashlib.sha256(
        "\n".join(sorted(task.task_id for task in tasks)).encode()
    ).hexdigest()
    manifest = {
        "status": "completed",
        "method": "QLoRA",
        "model_id": model_id,
        "resolved_revision": resolved_revision,
        "dataset_task_ids": [task.task_id for task in tasks],
        "task_id_fingerprint": fingerprint,
        "split": "train",
        "validation_used_for_updates": False,
        "test_accessed": False,
        "seed": seed,
        "lora": {"rank": 16, "alpha": 32, "dropout": 0.05},
        "training_seconds": elapsed,
        "metrics": trained.metrics,
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
    (destination / "adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return destination


def load_generator(model_id: str, adapter_path: str | Path | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for COBOL LLM inference")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path) if adapter_path else model_id
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
        device_map="auto",
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )
    model.eval()
    return model, tokenizer


def generate_suite(
    task: CobolTask,
    model,
    tokenizer,
    max_new_tokens: int = 256,
) -> str:
    import torch

    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": generation_prompt(task)}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.get_input_embeddings().weight.device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, encoded.shape[-1]:], skip_special_tokens=True)