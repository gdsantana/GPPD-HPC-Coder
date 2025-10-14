#!/usr/bin/env python3
import argparse
import os
import sys
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from dataset_loader import load_dataset_from_args

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Causal LM with QLoRA on hpcgroup/hpc-instruct (Cuda subset).")
    parser.add_argument("--model_name", type=str, required=True, help="Modelo base (ex.: deepseek-ai/deepseek-coder-1.3b-base ou ./meu-modelo)")
    parser.add_argument("--output_dir", type=str, default="./results", help="Diretório para checkpoints")
    parser.add_argument("--save_dir", type=str, default="./trained_model", help="Diretório onde salvar adapters/tokenizer")
    parser.add_argument("--max_length", type=int, default=512, help="Comprimento máximo de tokens")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas de treinamento")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size por dispositivo")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Passos de acumulação de gradiente")
    parser.add_argument("--use_evol_instruct", action="store_true", help="Adicionar dataset Evol-Instruct-Code-80k-v1 ao treinamento")
    parser.add_argument("--use_magicoder", action="store_true", help="Adicionar dataset Magicoder-OSS-Instruct-75K ao treinamento")
    parser.add_argument("--log_file", type=str, default=None, help="Caminho para arquivo de log. Se não especificado, usa apenas terminal")
    return parser.parse_args()

class DualLogger:
    """Logger que escreve simultaneamente no terminal e em arquivo."""
    def __init__(self, log_file=None):
        self.terminal = sys.stdout
        self.log_file = None
        
        if log_file:
            # Criar diretório se não existir
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Abrir arquivo em modo append
            self.log_file = open(log_file, 'a', encoding='utf-8')
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_file.write(f"\n{'='*80}\n")
            self.log_file.write(f"Início do treinamento: {timestamp}\n")
            self.log_file.write(f"{'='*80}\n\n")
            self.log_file.flush()
    
    def write(self, message):
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()
    
    def close(self):
        if self.log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_file.write(f"\n{'='*80}\n")
            self.log_file.write(f"Fim do treinamento: {timestamp}\n")
            self.log_file.write(f"{'='*80}\n\n")
            self.log_file.close()

def format_param_count(num_params):
    """Formata o número de parâmetros em formato legível (ex: 1.3B, 6.7B)."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.1f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.1f}M"
    else:
        return f"{num_params / 1e3:.1f}K"

def main():
    args = parse_args()
    
    # Configurar dual logging
    logger = DualLogger(args.log_file)
    sys.stdout = logger
    sys.stderr = logger

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
    
    # Obter contagem de parâmetros e criar diretório de salvamento
    num_params = model.num_parameters()
    param_str = format_param_count(num_params)
    print(f"Modelo carregado em 4 bits (QLoRA) com {num_params:,} parâmetros ({param_str}).")

    # ===== Dataset =====
    print("Carregando e preparando dataset...")
    
    # Adicionar atributos necessários para o dataset_loader
    if not hasattr(args, 'dataset_name'):
        args.dataset_name = "hpcgroup/hpc-instruct"
    if not hasattr(args, 'filter_language'):
        args.filter_language = "Cuda"
    
    # Usar o dataset_loader para suportar datasets opcionais
    tokenized_train = load_dataset_from_args(args, tokenizer)
    
    print(f"Dataset preparado com {len(tokenized_train)} exemplos")

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

    # Criar diretório de salvamento baseado no número de parâmetros
    save_dir = f"finetune_model-{param_str}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Salvar adaptadores LoRA
    lora_save_dir = os.path.join(save_dir, "lora_adapters")
    os.makedirs(lora_save_dir, exist_ok=True)
    print(f"Salvando adaptadores LoRA em {lora_save_dir}")
    trainer.save_model(lora_save_dir)
    
    # Salvar tokenizer no diretório principal
    tokenizer.save_pretrained(save_dir)
    
    # Tentar mesclar e salvar modelo completo
    print("Tentando mesclar adaptadores LoRA com modelo base...")
    try:
        merged_model = trainer.model.merge_and_unload()
        merged_save_dir = os.path.join(save_dir, "merged_model")
        os.makedirs(merged_save_dir, exist_ok=True)
        
        print(f"Salvando modelo mesclado em {merged_save_dir}")
        merged_model.save_pretrained(merged_save_dir)
        tokenizer.save_pretrained(merged_save_dir)
        print("Modelo mesclado salvo com sucesso")
    except Exception as e:
        print(f"Aviso: Não foi possível mesclar modelo: {e}")
        print("Use os adaptadores LoRA para inferência")
    
    print(f"\n{'='*80}")
    print(f"Treinamento concluído! Modelo salvo em: {save_dir}")
    print(f"  - Adaptadores LoRA: {lora_save_dir}")
    print(f"  - Tokenizer: {save_dir}")
    print(f"{'='*80}")
    
    # Fechar logger
    logger.close()
    sys.stdout = logger.terminal
    sys.stderr = logger.terminal

if __name__ == "__main__":
    main()
