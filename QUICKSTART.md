# 🚀 Quick Start - DeepSeek-Coder Fine-tuning

## Instalação de Dependências

```bash
# Instalar dependências
pip install -r requirements.txt
```

## Comando Principal de Treinamento

Use o script `finetune_with_args.py` com o seguinte comando:

```bash
python3 finetune_with_args.py \
  --model_name deepseek-ai/deepseek-coder-1.3b-instruct \
  --output_dir ./models/gppd-hpc-cuda-coder-instruct \
  --save_dir ./checkpoints/gppd-hpc-cuda-coder-instruct \
  --use_evol_instruct \
  --use_magicoder \
  --log_file ./logs/gppd-hpc-cuda-coder-instruct.log
```

## Parâmetros do Script

### Parâmetros Obrigatórios
- `--model_name`: Nome/caminho do modelo base (ex.: deepseek-ai/deepseek-coder-1.3b-instruct)

### Parâmetros Principais
- `--output_dir`: Diretório para checkpoints durante o treino (padrão: ./results)
- `--save_dir`: Diretório onde o modelo final será salvo (padrão: ./trained_model)
- `--log_file`: Caminho para arquivo de log (opcional)

### Datasets Adicionais
- `--use_evol_instruct`: Adiciona dataset Evol-Instruct-Code-80k-v1
- `--use_magicoder`: Adiciona dataset Magicoder-OSS-Instruct-75K

### Parâmetros de Configuração (opcionais)
- `--max_length`: Comprimento máximo de tokens (padrão: 512)
- `--epochs`: Número de épocas (padrão: 3)
- `--per_device_train_batch_size`: Batch size por dispositivo (padrão: 4)
- `--gradient_accumulation_steps`: Passos de acumulação de gradiente (padrão: 4)
- `--dataset_name`: Nome do dataset principal (padrão: hpcgroup/hpc-instruct)
- `--filter_language`: Filtrar por linguagem (padrão: Cuda)

## Exemplos de Uso

### Treinamento Básico (apenas HPC-Instruct)
```bash
python3 finetune_with_args.py \
  --model_name deepseek-ai/deepseek-coder-1.3b-instruct \
  --output_dir ./models/basic-cuda-coder \
  --save_dir ./checkpoints/basic-cuda-coder
```

### Treinamento com Configurações Customizadas
```bash
python3 finetune_with_args.py \
  --model_name deepseek-ai/deepseek-coder-1.3b-instruct \
  --output_dir ./models/custom-coder \
  --save_dir ./checkpoints/custom-coder \
  --epochs 5 \
  --max_length 1024 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --log_file ./logs/custom-training.log
```

### Treinamento Apenas com Evol-Instruct
```bash
python3 finetune_with_args.py \
  --model_name deepseek-ai/deepseek-coder-1.3b-instruct \
  --output_dir ./models/evol-coder \
  --save_dir ./checkpoints/evol-coder \
  --use_evol_instruct
```

## Estrutura de Saída

O modelo treinado será salvo com a seguinte estrutura:

```
finetune_model-{n_params}/
├── lora_adapters/          # Adaptadores LoRA (PEFT)
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
├── merged_model/           # Modelo completo mesclado
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

## Monitoramento

### Ver Logs em Tempo Real
```bash
tail -f ./logs/gppd-hpc-cuda-coder-instruct.log
```

### Monitorar GPU (se disponível)
```bash
python monitor_gpu.py
```

## Troubleshooting

### Erro de Memória (OOM)
1. Reduzir `--max_length 256`
2. Reduzir `--per_device_train_batch_size 1`
3. Aumentar `--gradient_accumulation_steps 8`

### Treinamento Muito Lento
1. Aumentar `--per_device_train_batch_size` (se tiver memória)
2. Reduzir `--gradient_accumulation_steps`
3. Usar modelo menor (1.3B ao invés de 6.7B)

### Erro de Dependências
```bash
pip install --upgrade transformers peft trl datasets bitsandbytes
```

## Recursos do Script

- ✅ Suporte a múltiplos datasets (HPC-Instruct, Evol-Instruct, Magicoder)
- ✅ Logging simultâneo no terminal e arquivo
- ✅ Quantização 8-bit para economia de memória
- ✅ LoRA (Low-Rank Adaptation) para fine-tuning eficiente
- ✅ Salvamento automático de adaptadores e modelo mesclado
- ✅ Timestamps e debug detalhado
