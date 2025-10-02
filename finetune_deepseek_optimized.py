#!/usr/bin/env python3
import argparse
import os
import torch
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def print_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune deepseek-coder-6.7B com otimizações para RTX 4090"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/deepseek-coder-6.7b-base",
        help="Nome/caminho do modelo base"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_deepseek",
        help="Diretório para checkpoints durante o treino"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./trained_deepseek",
        help="Diretório onde o modelo final será salvo"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
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
        default=1,
        help="Batch size por dispositivo (manter em 1 para modelo grande)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Passos de acumulação de gradiente (batch efetivo = batch_size * accumulation)"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="Rank do LoRA (8, 16, 32, 64)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=128,
        help="Alpha do LoRA (geralmente 2x o rank)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Usar Flash Attention 2 se disponível"
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        help="Usar packing de sequências para maior eficiência"
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
        "--max_samples",
        type=int,
        default=None,
        help="Limite de amostras para teste rápido (None = sem limite)"
    )
    return parser.parse_args()


def setup_model_and_tokenizer(args):
    logger.info(f"Carregando modelo: {args.model_name}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        "low_cpu_mem_usage": True,
    }
    
    if args.use_flash_attention:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 habilitado")
        except Exception as e:
            logger.warning(f"Flash Attention 2 não disponível: {e}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs
    )
    
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    
    logger.info(f"Modelo carregado: {model.num_parameters():,} parâmetros totais")
    print_memory_stats()
    
    return model, tokenizer


def setup_lora(model, args):
    logger.info(f"Configurando LoRA com r={args.lora_r}, alpha={args.lora_alpha}")
    
    target_modules = [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Parâmetros treináveis: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model


def load_and_prepare_dataset(args, tokenizer):
    logger.info(f"Carregando dataset: {args.dataset_name}")
    
    dataset = load_dataset(args.dataset_name)
    logger.info(f"Dataset original: {len(dataset['train'])} exemplos")
    
    if args.filter_language:
        dataset = dataset.filter(lambda x: x.get("language") == args.filter_language)
        logger.info(f"Após filtrar linguagem '{args.filter_language}': {len(dataset['train'])} exemplos")
    
    train_dataset = dataset["train"]
    
    if args.max_samples:
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
        logger.info(f"Limitado a {len(train_dataset)} amostras")
    
    def format_example(example):
        instruction = example.get('problem statement', '')
        response = example.get('solution', '')
        
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}{tokenizer.eos_token}"
        return {"text": text}
    
    logger.info("Formatando exemplos...")
    formatted_dataset = train_dataset.map(
        format_example,
        remove_columns=train_dataset.column_names,
        num_proc=4,
        desc="Formatando"
    )
    
    return formatted_dataset


def create_training_args(args):
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_bf16 = torch.cuda.is_bf16_supported()
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        warmup_steps=100,
        logging_steps=5,
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        fp16=not use_bf16,
        bf16=use_bf16,
        fp16_full_eval=False,
        bf16_full_eval=use_bf16,
        max_grad_norm=0.3,
        weight_decay=0.01,
        group_by_length=True,
        ddp_find_unused_parameters=False,
        report_to="none",
        push_to_hub=False,
        remove_unused_columns=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    logger.info(f"Configuração: BF16={use_bf16}, FP16={not use_bf16}")
    logger.info(f"Batch efetivo: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    
    return training_args


def main():
    args = parse_args()
    
    logger.info("="*80)
    logger.info("Fine-tuning deepseek-coder-6.7B otimizado para RTX 4090")
    logger.info("="*80)
    logger.info(f"Modelo: {args.model_name}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    logger.info(f"Batch size: {args.per_device_train_batch_size}, Accumulation: {args.gradient_accumulation_steps}")
    logger.info("="*80)
    
    clear_memory()
    
    model, tokenizer = setup_model_and_tokenizer(args)
    
    model = setup_lora(model, args)
    
    train_dataset = load_and_prepare_dataset(args, tokenizer)
    
    training_args = create_training_args(args)
    
    response_template = "### Response:"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=data_collator,
        max_seq_length=args.max_length,
        packing=args.packing,
        dataset_text_field="text",
    )
    
    model.config.use_cache = False
    
    logger.info("Iniciando treinamento...")
    print_memory_stats()
    
    try:
        trainer.train()
        logger.info("✓ Treinamento concluído com sucesso!")
    except Exception as e:
        logger.error(f"Erro durante treinamento: {e}")
        raise
    
    logger.info(f"Salvando modelo em: {args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    
    logger.info("✓ Modelo e tokenizer salvos")
    
    logger.info("Tentando salvar modelo mesclado...")
    try:
        model.config.use_cache = True
        merged_model = model.merge_and_unload()
        
        merged_save_dir = os.path.join(args.save_dir, "merged_model")
        os.makedirs(merged_save_dir, exist_ok=True)
        
        merged_model.save_pretrained(
            merged_save_dir,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        tokenizer.save_pretrained(merged_save_dir)
        
        logger.info(f"✓ Modelo mesclado salvo em: {merged_save_dir}")
    except Exception as e:
        logger.warning(f"Não foi possível salvar modelo mesclado: {e}")
        logger.info("Use os adaptadores LoRA para inferência")
    
    logger.info("="*80)
    logger.info("Fine-tuning completo!")
    logger.info(f"Adaptadores LoRA: {args.save_dir}")
    logger.info("="*80)
    
    print_memory_stats()


if __name__ == "__main__":
    main()
