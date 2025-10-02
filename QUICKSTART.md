# 🚀 Quick Start - DeepSeek-Coder-6.7B Fine-tuning

## Setup Rápido

```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

## Verificar Compatibilidade

```bash
python check_compatibility.py
```

## Treinar (Configuração Recomendada)

```bash
python finetune_deepseek_optimized.py \
  --model_name deepseek-ai/deepseek-coder-6.7b-base \
  --output_dir ./results \
  --save_dir ./model_trained \
  --epochs 3 \
  --max_length 1024 \
  --lora_r 64 \
  --lora_alpha 128 \
  --gradient_accumulation_steps 16 \
  --use_flash_attention \
  --packing
```

## Teste Rápido (100 amostras)

```bash
python finetune_deepseek_optimized.py \
  --model_name deepseek-ai/deepseek-coder-6.7b-base \
  --max_samples 100 \
  --epochs 1
```

## Monitorar GPU

```bash
python monitor_gpu.py
```

## Inferência

### Com adaptadores LoRA
```bash
python inference_example.py \
  --adapter_path ./model_trained \
  --instruction "Write a CUDA kernel for vector addition"
```

### Modo interativo
```bash
python inference_example.py \
  --adapter_path ./model_trained \
  --interactive
```

### Com modelo mesclado
```bash
python inference_example.py \
  --merged_model_path ./model_trained/merged_model \
  --interactive
```

## Configurações Alternativas

### Conservative (menos memória, ~18GB)
```bash
python finetune_deepseek_optimized.py \
  --max_length 512 \
  --lora_r 32 \
  --lora_alpha 64 \
  --gradient_accumulation_steps 8
```

### High Quality (mais memória, ~22GB)
```bash
python finetune_deepseek_optimized.py \
  --max_length 2048 \
  --lora_r 128 \
  --lora_alpha 256 \
  --gradient_accumulation_steps 32 \
  --use_flash_attention \
  --packing
```

## Estrutura de Arquivos Criados

```
TCC/
├── finetune_deepseek_optimized.py  # Script principal de treinamento
├── requirements.txt                 # Dependências Python
├── setup_environment.sh             # Setup automático
├── check_compatibility.py           # Verificação de sistema
├── monitor_gpu.py                   # Monitor de memória GPU
├── inference_example.py             # Script de inferência
├── README_DEEPSEEK_FINETUNE.md     # Documentação completa
└── QUICKSTART.md                    # Este arquivo
```

## Troubleshooting

### OOM Error
1. Reduzir `--max_length 512`
2. Reduzir `--lora_r 32`
3. Aumentar `--gradient_accumulation_steps 32`

### Muito Lento
1. Adicionar `--use_flash_attention`
2. Adicionar `--packing`
3. Aumentar `--per_device_train_batch_size` (se tiver memória)

### Flash Attention Error
```bash
pip install flash-attn --no-build-isolation
```
Ou remover `--use_flash_attention`

## Ver Documentação Completa

```bash
cat README_DEEPSEEK_FINETUNE.md
```
