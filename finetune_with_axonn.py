#!/usr/bin/env python3
"""
Script de fine-tuning com AxoNN para treinamento distribuído multi-nó.
Utiliza paralelização de tensor e dados para escalar o treinamento.
"""

import argparse
import os
import sys
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator, AxoNNPlugin
import axonn
from dataset_loader import load_dataset_from_args
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Causal LM com LoRA usando AxoNN para treinamento distribuído."
    )
    
    # Argumentos do modelo
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Nome/caminho do modelo base do HuggingFace (ex.: deepseek-ai/deepseek-coder-6.7b-base)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_axonn",
        help="Diretório para checkpoints durante o treino"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Diretório onde o modelo completo será salvo"
    )
    
    # Argumentos de treinamento
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
        default=2,
        help="Batch size por dispositivo"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Passos de acumulação de gradiente"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Taxa de aprendizado"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Número de passos de warmup"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Frequência de logging"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Frequência de salvamento de checkpoints"
    )
    
    # Argumentos de dataset
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
    
    # Argumentos de LoRA
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="Rank do LoRA"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Alpha do LoRA"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout do LoRA"
    )
    
    # Argumentos específicos do AxoNN
    parser.add_argument(
        "--tensor_parallelism",
        type=int,
        default=1,
        help="Grau de paralelismo de tensor (G_intra_depth)"
    )
    parser.add_argument(
        "--data_parallelism",
        type=int,
        default=1,
        help="Grau de paralelismo de dados"
    )
    
    # Argumentos de logging
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Caminho para arquivo de log"
    )
    
    return parser.parse_args()


class DualLogger:
    """Logger que escreve simultaneamente no terminal e em arquivo."""
    def __init__(self, log_file=None):
        self.terminal = sys.stdout
        self.log_file = None
        
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            self.log_file = open(log_file, 'a', encoding='utf-8')
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_file.write(f"\n{'='*80}\n")
            self.log_file.write(f"Início do treinamento com AxoNN: {timestamp}\n")
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
            self.log_file.write(f"Fim do treinamento com AxoNN: {timestamp}\n")
            self.log_file.write(f"{'='*80}\n\n")
            self.log_file.close()


def format_param_count(num_params):
    """Formata o número de parâmetros em formato legível."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.1f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.1f}M"
    else:
        return f"{num_params / 1e3:.1f}K"


def train_epoch(model, dataloader, optimizer, accelerator, epoch, args):
    """Treina uma época."""
    model.train()
    total_loss = 0
    step = 0
    
    for batch_idx, batch in enumerate(dataloader):
        with accelerator.accumulate(model):
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            step += 1
            
            # Logging
            if step % args.logging_steps == 0:
                avg_loss = total_loss / step
                logger.info(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}")
            
            # Checkpoint
            if step % args.save_steps == 0 and accelerator.is_main_process:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch{epoch}-step{step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                accelerator.save_state(checkpoint_dir)
                logger.info(f"Checkpoint salvo em: {checkpoint_dir}")
    
    return total_loss / step


def main():
    args = parse_args()
    
    # Configurar dual logging
    logger_dual = DualLogger(args.log_file)
    sys.stdout = logger_dual
    sys.stderr = logger_dual
    
    logger.info("="*80)
    logger.info("Iniciando fine-tuning com AxoNN")
    logger.info("="*80)
    logger.info(f"Modelo: {args.model_name}")
    logger.info(f"Paralelismo de tensor: {args.tensor_parallelism}")
    logger.info(f"Paralelismo de dados: {args.data_parallelism}")
    logger.info(f"Diretório de salvamento: {args.save_dir}")
    logger.info("="*80)
    
    # ===== Configuração AxoNN =====
    logger.info("\n[1/6] Configurando AxoNN...")
    
    # Criar plugin AxoNN
    axonn_plugin = AxoNNPlugin(
        G_intra_depth=args.tensor_parallelism,
        G_intra_col=1,
        G_intra_row=1
    )
    
    # Inicializar Accelerator com AxoNN
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        axonn_plugin=axonn_plugin,
        log_with="none"
    )
    
    logger.info(f"✓ AxoNN configurado - Processo {accelerator.process_index}/{accelerator.num_processes}")
    
    # ===== Carregamento do modelo =====
    logger.info(f"\n[2/6] Carregando modelo: {args.model_name}")
    
    # Carregar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("✓ Pad token configurado como EOS token")
    
    # Carregar modelo com paralelização AxoNN
    with axonn.models.transformers.parallelize(args.model_name):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    
    num_params = model.num_parameters()
    param_str = format_param_count(num_params)
    logger.info(f"✓ Modelo carregado: {num_params:,} parâmetros ({param_str})")
    
    # ===== Configuração LoRA =====
    logger.info(f"\n[3/6] Configurando LoRA...")
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✓ LoRA configurado - Parâmetros treináveis: {format_param_count(trainable_params)}")
    model.print_trainable_parameters()
    
    # ===== Carregamento do dataset =====
    logger.info(f"\n[4/6] Carregando dataset...")
    
    dataset = load_dataset_from_args(args, tokenizer)
    logger.info(f"✓ Dataset carregado: {len(dataset)} exemplos")
    
    # Preparar dataloader
    def collate_fn(examples):
        texts = [ex["text"] for ex in examples]
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=False
    )
    
    # ===== Configuração do otimizador =====
    logger.info(f"\n[5/6] Configurando otimizador...")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Preparar com Accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    logger.info(f"✓ Otimizador configurado - LR: {args.learning_rate}")
    
    # ===== Treinamento =====
    logger.info(f"\n[6/6] Iniciando treinamento...")
    logger.info(f"Épocas: {args.epochs}")
    logger.info(f"Batch size por dispositivo: {args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"Total de steps por época: {len(dataloader)}")
    logger.info("="*80)
    
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*80}")
        logger.info(f"ÉPOCA {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*80}")
        
        avg_loss = train_epoch(model, dataloader, optimizer, accelerator, epoch + 1, args)
        
        logger.info(f"Época {epoch + 1} concluída - Loss médio: {avg_loss:.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("Treinamento concluído!")
    logger.info("="*80)
    
    # ===== Salvar modelo =====
    if accelerator.is_main_process:
        logger.info(f"\nSalvando modelo em: {args.save_dir}")
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Salvar adaptadores LoRA
        lora_save_dir = os.path.join(args.save_dir, "lora_adapters")
        os.makedirs(lora_save_dir, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(lora_save_dir)
        tokenizer.save_pretrained(lora_save_dir)
        logger.info(f"✓ Adaptadores LoRA salvos em: {lora_save_dir}")
        
        # Tentar mesclar modelo
        try:
            logger.info("Mesclando adaptadores LoRA com modelo base...")
            merged_model = unwrapped_model.merge_and_unload()
            merged_save_dir = os.path.join(args.save_dir, "merged_model")
            os.makedirs(merged_save_dir, exist_ok=True)
            merged_model.save_pretrained(merged_save_dir)
            tokenizer.save_pretrained(merged_save_dir)
            logger.info(f"✓ Modelo mesclado salvo em: {merged_save_dir}")
        except Exception as e:
            logger.warning(f"Não foi possível mesclar modelo: {e}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Modelo salvo com sucesso!")
        logger.info(f"  - Adaptadores LoRA: {lora_save_dir}")
        logger.info(f"  - Modelo mesclado: {merged_save_dir}")
        logger.info(f"{'='*80}")
    
    # Aguardar todos os processos
    accelerator.wait_for_everyone()
    
    logger.info("\nProcesso de fine-tuning com AxoNN completado!")
    
    # Fechar logger
    logger_dual.close()
    sys.stdout = logger_dual.terminal
    sys.stderr = logger_dual.terminal


if __name__ == "__main__":
    main()
