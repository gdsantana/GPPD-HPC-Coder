#!/usr/bin/env python3
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Causal LM with QLoRA on hpcgroup/hpc-instruct (Cuda subset).")
    parser.add_argument("--model_name", type=str, required=True, help="Modelo base (ex.: deepseek-ai/deepseek-coder-1.3b-base ou ./meu-modelo)")
    parser.add_argument("--output_dir", type=str, default="./results", help="Diretório para checkpoints")
    parser.add_argument("--save_dir", type=str, default="./trained_model", help="Diretório onde salvar adapters/tokenizer")
    parser.add_argument("--max_length", type=int, default=512, help="Comprimento máximo de tokens")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas de treinamento")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size por dispositivo")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Passos de acumulação de gradiente")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Carregando modelo base: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Configuração para quantização em 4 bits (necessário para QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    print("Modelo carregado em 4 bits (QLoRA).")

    # ===== Dataset =====
    dataset = load_dataset("hpcgroup/hpc-instruct")
    cuda_dataset = dataset.filter(lambda ex: ex.get("language") == "Cuda")
    train_dataset = cuda_dataset["train"]

    def format_example(example):
        return {
            "text": f"Instruction: {example['problem statement']}\nResponse: {example['solution']}"
        }

    formatted_train = train_dataset.map(format_example, remove_columns=train_dataset.column_names)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length
        )

    tokenized_train = formatted_train.map(tokenize_function, batched=True)

    # ===== QLoRA Config =====
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]  # tipicamente usado em QLoRA
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_8bit",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        push_to_hub=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train,
        peft_config=lora_config,
        args=training_args,
    )

    print("Iniciando treinamento...")
    trainer.train()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Salvando modelo treinado em {args.save_dir}")
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    print("Treinamento concluído e modelo salvo.")

if __name__ == "__main__":
    main()
