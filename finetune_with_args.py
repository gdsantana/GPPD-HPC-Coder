#!/usr/bin/env python3
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

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
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = args.model_name

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
    print(f"DEBUG: Modelo carregado com {model.num_parameters()} parametros")

    # ===== Dataset =====
    print("Carregando dataset hpcgroup/hpc-instruct e filtrando linguagem = 'Cuda'...")
    dataset = load_dataset("hpcgroup/hpc-instruct")
    print(f"DEBUG: Dataset original carregado com {len(dataset['train'])} exemplos de treino")
    
    cuda_dataset = dataset.filter(lambda example: example.get("language") == "Cuda")
    print(f"DEBUG: Filtrado para linguagem CUDA")

    # Usar TODO o dataset de treinamento (subconjunto CUDA completo)
    train_dataset = cuda_dataset["train"]
    print(f"DEBUG: Dataset CUDA final tem {len(train_dataset)} exemplos")

    # Formatação -> campo 'text' para o treinador
    def format_example(example):
        # Mantido conforme o script original: 'problem statement' e 'solution'
        # Ajuste aqui se as chaves do dataset forem diferentes no seu ambiente.
        return {
            "text": f"Instruction: {example['problem statement']}\nResponse: {example['solution']}"
        }

    print("Formatando exemplos...")
    formatted_train = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    print(f"DEBUG: Exemplos formatados com sucesso")

    # Tokenização
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length
        )

    print("Tokenizando dataset...")
    tokenized_train = formatted_train.map(tokenize_function, batched=True)
    print(f"DEBUG: Dataset tokenizado com max_length={args.max_length}")

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
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"Salvando modelo completo fine-tuned e tokenizer em: {save_dir}")
    print("DEBUG: Iniciando processo de salvamento do modelo")
    
    # Salvar o modelo com adaptadores LoRA
    print("DEBUG: Salvando modelo com adaptadores LoRA...")
    trainer.save_model(save_dir)
    print("DEBUG: Modelo com adaptadores LoRA salvo")
    
    # Salvar tokenizer
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
    
    print(f"Modelo e tokenizer salvos com sucesso em {save_dir}")
    print("Processo de fine-tuning completado!")
    print("DEBUG: Todos os arquivos foram salvos")
    print("DEBUG: Processo finalizado com sucesso")

if __name__ == "__main__":
    main()
