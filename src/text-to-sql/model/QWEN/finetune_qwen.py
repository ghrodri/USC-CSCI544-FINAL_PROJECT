import json
from pathlib import Path
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

IR_DATA_DIR = Path("./src/text-to-sql/ir-data")
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = Path("./src/text-to-sql/model/QWEN")

SAMPLE_TRAIN_SIZE = 100_000
MAX_LENGTH = 128
BATCH_SIZE = 16
GRAD_ACCUM = 2
LR = 2e-4
EPOCHS = 10


def build_prompt(example):
    q = example["question"]
    ir = example["ir"]
    ir_json = json.dumps(ir, ensure_ascii=False)
    return f"Question: {q}\nIR:\n{ir_json}"


def load_ir_dataset():
    train_files = list((IR_DATA_DIR / "train").glob("*_ir.jsonl"))
    val_files = list((IR_DATA_DIR / "val").glob("*_ir.jsonl"))

    if not train_files:
        raise RuntimeError(f"No train *_ir.jsonl files found in {IR_DATA_DIR/'train'}")
    if not val_files:
        raise RuntimeError(f"No val *_ir.jsonl files found in {IR_DATA_DIR/'val'}")

    train_ds = load_dataset(
        "json",
        data_files=[str(p) for p in train_files],
        split="train",
    )
    val_ds = load_dataset(
        "json",
        data_files=[str(p) for p in val_files],
        split="train",
    )

    if len(train_ds) > SAMPLE_TRAIN_SIZE:
        train_ds = train_ds.shuffle(seed=42).select(range(SAMPLE_TRAIN_SIZE))
    else:
        train_ds = train_ds.shuffle(seed=42)

    return DatasetDict({"train": train_ds, "validation": val_ds})


def tokenize_function(example, tokenizer):
    text = example["prompt"]
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def main():
    print("[STEP] Loading IR dataset...")
    dset = load_ir_dataset()

    print("[STEP] Building prompts...")
    dset = dset.map(lambda x: {"prompt": build_prompt(x)})

    print("[STEP] Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    print("[STEP] Applying LoRA...")
    model = get_peft_model(base_model, lora_cfg)

    print("[STEP] Tokenizing dataset...")
    tokenized = dset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dset["train"].column_names,
    )

    OUTPUT_DIR_PATH = Path(OUTPUT_DIR)
    OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR_PATH),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=50,
        save_steps=1000,
        save_total_limit=2,
        fp16=True,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving LoRA adapter and tokenizer...")
    trainer.save_model(str(OUTPUT_DIR_PATH))
    tokenizer.save_pretrained(str(OUTPUT_DIR_PATH))

    print("Training finished.")

if __name__ == "__main__":
    main()