#!/usr/bin/env python3
import argparse
import os
import json
import logging
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import psutil
from typing import Optional, Dict, Any
import sys

class ProgressCallback(TrainerCallback):
    """Callback personalizado para monitorar progresso"""
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            logging.info(f"Step {state.global_step}: {logs}")

def setup_logging(log_level: str = "INFO"):
    """Configura logging detalhado"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('finetune_qlora.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def validate_environment():
    """Valida ambiente e recursos disponíveis"""
    if not torch.cuda.is_available():
        logging.error("CUDA não disponível. QLoRA requer GPU!")
        return False
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    ram_memory = psutil.virtual_memory().total / 1024**3
    
    logging.info(f"GPU Memory: {gpu_memory:.1f}GB")
    logging.info(f"RAM Memory: {ram_memory:.1f}GB")
    
    if gpu_memory < 6:
        logging.warning("GPU com pouca memória (<6GB) para QLoRA. Pode haver problemas")
    
    return True

def auto_batch_size_qlora(available_memory_gb: float) -> int:
    """Calcula batch size para QLoRA baseado na memória"""
    # QLoRA é mais eficiente em memória
    if available_memory_gb >= 24:
        return 8
    elif available_memory_gb >= 16:
        return 6
    elif available_memory_gb >= 12:
        return 4
    elif available_memory_gb >= 8:
        return 2
    else:
        return 1

def load_config_file(config_path: str) -> Dict[str, Any]:
    """Carrega configuração de arquivo JSON"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Arquivo de config {config_path} não encontrado")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Erro ao ler config JSON: {e}")
        return {}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Causal LM with QLoRA - Versão Aprimorada"
    )
    
    # Argumentos básicos
    parser.add_argument("--model_name", type=str, required=True,
                       help="Nome/caminho do modelo base")
    parser.add_argument("--output_dir", type=str, default="./results_qlora",
                       help="Diretório para checkpoints")
    parser.add_argument("--save_dir", type=str, default="./trained_model_qlora",
                       help="Diretório para modelo final")
    parser.add_argument("--config", type=str,
                       help="Arquivo JSON com configurações")
    
    # Parâmetros de dados
    parser.add_argument("--dataset_name", type=str, default="hpcgroup/hpc-instruct",
                       help="Nome do dataset")
    parser.add_argument("--dataset_subset", type=str, default="Cuda",
                       help="Subconjunto do dataset")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Comprimento máximo de tokens")
    parser.add_argument("--validation_split", type=float, default=0.1,
                       help="Porcentagem para validação")
    
    # Parâmetros de treinamento
    parser.add_argument("--epochs", type=int, default=3,
                       help="Número de épocas")
    parser.add_argument("--per_device_train_batch_size", type=int, default=0,
                       help="Batch size (0 = automático)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Passos de acumulação")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Taxa de aprendizado")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                       help="Ratio de warmup")
    
    # Parâmetros QLoRA específicos
    parser.add_argument("--lora_r", type=int, default=64,
                       help="Rank do LoRA (maior para QLoRA)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="Alpha do LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="Dropout do LoRA")
    parser.add_argument("--target_modules", type=str, nargs="+", 
                       default=["q_proj", "v_proj", "k_proj", "o_proj"],
                       help="Módulos alvo para LoRA")
    
    # Parâmetros de quantização
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                       help="Usar quantização 4-bit")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16"],
                       help="Tipo de dados para computação")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4",
                       choices=["fp4", "nf4"],
                       help="Tipo de quantização")
    
    # Controle de treinamento
    parser.add_argument("--resume_from_checkpoint", type=str,
                       help="Checkpoint para continuar")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                       help="Paciência para early stopping")
    parser.add_argument("--save_steps", type=int, default=100,
                       help="Frequência de salvamento")
    parser.add_argument("--eval_steps", type=int, default=50,
                       help="Frequência de avaliação")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Frequência de logging")
    
    # Outros
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Nível de logging")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed para reprodutibilidade")
    
    return parser.parse_args()

def prepare_dataset(args, tokenizer):
    """Prepara dataset com validação para QLoRA"""
    try:
        logging.info(f"Carregando dataset: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name)
        
        if args.dataset_subset:
            logging.info(f"Filtrando por linguagem: {args.dataset_subset}")
            filtered_dataset = dataset.filter(
                lambda example: example.get("language") == args.dataset_subset
            )
        else:
            filtered_dataset = dataset
        
        train_dataset = filtered_dataset["train"]
        logging.info(f"Dataset carregado: {len(train_dataset)} exemplos")
        
        # Split de validação
        if args.validation_split > 0:
            split_dataset = train_dataset.train_test_split(
                test_size=args.validation_split,
                seed=args.seed
            )
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
            logging.info(f"Split - Train: {len(train_dataset)}, Val: {len(eval_dataset)}")
        else:
            eval_dataset = None
        
        # Formatação
        def format_example(example):
            return {
                "text": f"Instruction: {example['problem statement']}\nResponse: {example['solution']}"
            }
        
        logging.info("Formatando exemplos...")
        formatted_train = train_dataset.map(
            format_example,
            remove_columns=train_dataset.column_names,
            desc="Formatando treino"
        )
        
        formatted_eval = None
        if eval_dataset:
            formatted_eval = eval_dataset.map(
                format_example,
                remove_columns=eval_dataset.column_names,
                desc="Formatando validação"
            )
        
        # Tokenização
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=args.max_length
            )
        
        logging.info("Tokenizando dataset...")
        tokenized_train = formatted_train.map(
            tokenize_function,
            batched=True,
            desc="Tokenizando treino"
        )
        
        tokenized_eval = None
        if formatted_eval:
            tokenized_eval = formatted_eval.map(
                tokenize_function,
                batched=True,
                desc="Tokenizando validação"
            )
        
        return tokenized_train, tokenized_eval
        
    except Exception as e:
        logging.error(f"Erro ao preparar dataset: {e}")
        raise

def load_model_and_tokenizer_qlora(args):
    """Carrega modelo e tokenizer com configuração QLoRA"""
    try:
        logging.info(f"Carregando tokenizer: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logging.info("Pad token definido como EOS token")
        
        # Configuração de quantização 4-bit
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype
        )
        
        logging.info(f"Carregando modelo com QLoRA: {args.model_name}")
        logging.info(f"Quantização: {args.bnb_4bit_quant_type}, Compute dtype: {args.bnb_4bit_compute_dtype}")
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        logging.info("Modelo QLoRA carregado com sucesso")
        return model, tokenizer
        
    except Exception as e:
        logging.error(f"Erro ao carregar modelo QLoRA: {e}")
        raise

def create_qlora_config(args, tokenized_train, tokenized_eval):
    """Cria configurações específicas para QLoRA"""
    
    # Batch size automático
    batch_size = args.per_device_train_batch_size
    if batch_size == 0:
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            batch_size = auto_batch_size_qlora(gpu_memory)
        else:
            batch_size = 1
        logging.info(f"Batch size automático para QLoRA: {batch_size}")
    
    # Configuração LoRA otimizada para QLoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    
    # Argumentos de treinamento otimizados para QLoRA
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_8bit",  # Otimizador específico para QLoRA
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        save_total_limit=3,
        evaluation_strategy="steps" if tokenized_eval else "no",
        eval_steps=args.eval_steps if tokenized_eval else None,
        load_best_model_at_end=True if tokenized_eval else False,
        metric_for_best_model="eval_loss" if tokenized_eval else None,
        fp16=False,  # QLoRA usa bfloat16 internamente
        bf16=True if args.bnb_4bit_compute_dtype == "bfloat16" else False,
        push_to_hub=False,
        report_to="none",
        seed=args.seed,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,  # Economiza memória
        remove_unused_columns=False,
    )
    
    return lora_config, training_args

def main():
    args = parse_args()
    
    # Carregar configuração
    if args.config:
        config = load_config_file(args.config)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # Setup
    setup_logging(args.log_level)
    torch.manual_seed(args.seed)
    
    logging.info("=== Iniciando QLoRA Fine-tuning Aprimorado ===")
    logging.info(f"Argumentos: {vars(args)}")
    
    # Validar ambiente
    if not validate_environment():
        logging.error("Ambiente não adequado para QLoRA")
        sys.exit(1)
    
    try:
        # Carregar modelo e tokenizer
        model, tokenizer = load_model_and_tokenizer_qlora(args)
        
        # Preparar dataset
        tokenized_train, tokenized_eval = prepare_dataset(args, tokenizer)
        
        # Criar configurações QLoRA
        lora_config, training_args = create_qlora_config(
            args, tokenized_train, tokenized_eval
        )
        
        logging.info("Configurações QLoRA:")
        logging.info(f"LoRA - r: {lora_config.r}, alpha: {lora_config.lora_alpha}")
        logging.info(f"Target modules: {lora_config.target_modules}")
        logging.info(f"Batch size: {training_args.per_device_train_batch_size}")
        logging.info(f"Learning rate: {training_args.learning_rate}")
        
        # Callbacks
        callbacks = [ProgressCallback()]
        if tokenized_eval and args.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
            )
        
        # Criar trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            peft_config=lora_config,
            args=training_args,
            callbacks=callbacks,
        )
        
        # Treinamento
        logging.info("=== Iniciando Treinamento QLoRA ===")
        
        if args.resume_from_checkpoint:
            logging.info(f"Resumindo de checkpoint: {args.resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train()
        
        logging.info("=== Treinamento QLoRA Concluído ===")
        
        # Salvar modelo
        os.makedirs(args.save_dir, exist_ok=True)
        logging.info(f"Salvando modelo QLoRA em: {args.save_dir}")
        
        trainer.save_model(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        
        # Salvar configurações
        config_path = os.path.join(args.save_dir, "qlora_config.json")
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        logging.info("=== QLoRA Finalizado com Sucesso ===")
        
    except KeyboardInterrupt:
        logging.info("Treinamento QLoRA interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Erro durante treinamento QLoRA: {e}")
        raise

if __name__ == "__main__":
    main()
