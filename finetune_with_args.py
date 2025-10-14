#!/usr/bin/env python3
import argparse
import os
import sys
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from dataset_loader import DatasetConfig, DatasetLoader, load_dataset_from_args

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Causal LM with LoRA on hpcgroup/hpc-instruct (Cuda subset)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Nome/caminho do modelo base (ex.: deepseek-ai/deepseek-coder-1.3b-base ou ./meu-modelo)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Diretório para checkpoints durante o treino"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./trained_model",
        help="Diretório onde o modelo final (adapters/tokenizer) será salvo"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Comprimento máximo de tokens por exemplo"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Número de épocas de treinamento"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size por dispositivo"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Passos de acumulação de gradiente"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="hpcgroup/hpc-instruct",
        help="Nome do dataset no HuggingFace"
    )
    parser.add_argument(
        "--filter_language",
        type=str,
        default="Cuda",
        help="Filtrar dataset por linguagem específica"
    )
    parser.add_argument(
        "--use_evol_instruct",
        action="store_true",
        help="Adicionar dataset Evol-Instruct-Code-80k-v1 ao treinamento"
    )
    parser.add_argument(
        "--use_magicoder",
        action="store_true",
        help="Adicionar dataset Magicoder-OSS-Instruct-75K ao treinamento"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Caminho para arquivo de log. Se não especificado, usa apenas terminal"
    )
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
    model_name = args.model_name
    
    # Configurar dual logging
    logger = DualLogger(args.log_file)
    sys.stdout = logger
    sys.stderr = logger

    print(f"DEBUG: Iniciando fine-tuning com modelo: {model_name}")
    print(f"DEBUG: Parametros - epochs: {args.epochs}, batch_size: {args.per_device_train_batch_size}")
    print(f"DEBUG: Output dir: {args.output_dir}, Save dir: {args.save_dir}")
    
    print(f"Carregando tokenizer e modelo base: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Adicionar pad_token se não existir
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("DEBUG: Pad token configurado como EOS token")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_8bit=True  # requer bitsandbytes
    )
    print("Modelo e tokenizer carregados com sucesso.")
    
    # Obter contagem de parâmetros e criar diretório de salvamento
    num_params = model.num_parameters()
    param_str = format_param_count(num_params)
    print(f"DEBUG: Modelo carregado com {num_params:,} parametros ({param_str})")

    # ===== Dataset =====
    print("Carregando e preparando dataset...")
    
    # Verificar se há datasets adicionais
    has_additional = (hasattr(args, 'use_evol_instruct') and args.use_evol_instruct) or \
                     (hasattr(args, 'use_magicoder') and args.use_magicoder)
    
    if has_additional:
        # Usar função integrada que suporta múltiplos datasets
        # Nota: load_dataset_from_args usa o formato padrão "### Instruction:\n{instruction}\n\n### Response:\n{response}"
        print("AVISO: Usando formato padrão devido a datasets adicionais")
        tokenized_train = load_dataset_from_args(args, tokenizer)
    else:
        # Usar formato customizado apenas para dataset único
        loader = DatasetLoader(tokenizer)
        
        dataset_config = DatasetConfig(
            name=args.dataset_name,
            instruction_column="problem statement",
            response_column="solution",
            filter_language=args.filter_language
        )
        
        # Formato customizado para este script (sem ###)
        format_template = "Instruction: {instruction}\nResponse: {response}"
        tokenized_train = loader.load_single_dataset(dataset_config, format_template)
        
        if tokenized_train is None:
            raise ValueError("Falha ao carregar dataset")
    
    print(f"DEBUG: Dataset preparado com {len(tokenized_train)} exemplos")

    # ===== LoRA / QLoRA =====
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print("Configuração LoRA criada.")
    print(f"DEBUG: LoRA config - r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")

    # ===== Training Args =====
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
        save_steps=100,           # menos frequente para reduzir I/O
        save_total_limit=3,
        fp16=True,
        push_to_hub=False,
        report_to="none",
    )
    print("Argumentos de treinamento configurados.")

    # ===== Trainer =====
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train,
        peft_config=lora_config,
        args=training_args,
    )
    print(f"DEBUG: Trainer configurado com {len(tokenized_train)} exemplos de treino")

    print("Iniciando treinamento...")
    trainer.train()
    print("Treinamento concluído com sucesso!")
    print("DEBUG: Processo de treinamento finalizado")

    # ===== Salvar modelo final completo (com adaptações LoRA mescladas) e tokenizer =====
    # Criar diretório de salvamento baseado no número de parâmetros
    save_dir = f"finetune_model-{param_str}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Salvando modelo completo fine-tuned e tokenizer em: {save_dir}")
    print("DEBUG: Iniciando processo de salvamento do modelo")
    
    # Salvar o modelo com adaptadores LoRA
    lora_save_dir = os.path.join(save_dir, "lora_adapters")
    os.makedirs(lora_save_dir, exist_ok=True)
    print(f"DEBUG: Salvando modelo com adaptadores LoRA em {lora_save_dir}...")
    trainer.save_model(lora_save_dir)
    print("DEBUG: Modelo com adaptadores LoRA salvo")
    
    # Salvar tokenizer no diretório principal
    print("DEBUG: Salvando tokenizer...")
    tokenizer.save_pretrained(save_dir)
    print("DEBUG: Tokenizer salvo")
    
    # Mesclar e salvar modelo completo (opcional - para usar sem PEFT)
    print("DEBUG: Mesclando adaptadores LoRA com modelo base...")
    try:
        # Obter o modelo base com adaptadores mesclados
        merged_model = trainer.model.merge_and_unload()
        merged_save_dir = os.path.join(save_dir, "merged_model")
        os.makedirs(merged_save_dir, exist_ok=True)
        
        print(f"DEBUG: Salvando modelo mesclado em {merged_save_dir}")
        merged_model.save_pretrained(merged_save_dir)
        tokenizer.save_pretrained(merged_save_dir)
        print("DEBUG: Modelo mesclado salvo com sucesso")
    except Exception as e:
        print(f"DEBUG: Erro ao mesclar modelo: {e}")
        print("DEBUG: Modelo com adaptadores LoRA foi salvo normalmente")
    
    print(f"\n{'='*80}")
    print(f"Modelo e tokenizer salvos com sucesso em: {save_dir}")
    print(f"  - Adaptadores LoRA: {lora_save_dir}")
    print(f"  - Modelo mesclado: {merged_save_dir}")
    print(f"  - Tokenizer: {save_dir}")
    print(f"{'='*80}")
    print("Processo de fine-tuning completado!")
    print("DEBUG: Todos os arquivos foram salvos")
    print("DEBUG: Processo finalizado com sucesso")
    
    # Fechar logger
    logger.close()
    sys.stdout = logger.terminal
    sys.stderr = logger.terminal

if __name__ == "__main__":
    main()
