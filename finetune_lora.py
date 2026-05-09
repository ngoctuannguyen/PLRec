import os
import torch
import sys
cwd = sys.path.pop(0)
from datasets import load_dataset
sys.path.insert(0, cwd)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def main():
    dataset_code = "games"
    data_path = f"data/{dataset_code}/lora_train.jsonl"
    output_dir = f"experiments/lora/{dataset_code}_qwen2.5_7b"
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    if not os.path.exists(data_path):
        print(f"Error: Training data not found at {data_path}")
        return

    # 1. Load Dataset & convert to chat format for completion_only_loss
    print("Loading dataset...")
    raw_dataset = load_dataset("json", data_files={"train": data_path})
    
    def to_chat_messages(example):
        """Convert instruction/output format to chat messages format.
        This allows SFTTrainer to identify the assistant's response
        and only compute loss on that part (completion_only_loss)."""
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]}
        ]
        return {"messages": messages}
    
    train_data = raw_dataset["train"].map(to_chat_messages, remove_columns=["instruction", "output"])
    print(f"Dataset: {len(train_data)} samples, columns: {train_data.column_names}")

    # 2. Configure Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Load Model in 4-bit
    print("Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 5. Training Arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="none",
        max_length=1024,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=True,  # Only compute loss on assistant response (the answer letter)
    )

    # 6. Trainer (no formatting_func needed — using chat messages format)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # 7. Start Training
    print("Starting training...")
    trainer.train()
    
    # Save final adapter
    trainer.model.save_pretrained(os.path.join(output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_adapter"))
    print(f"Training complete! LoRA adapter saved to {output_dir}/final_adapter")

if __name__ == "__main__":
    main()
