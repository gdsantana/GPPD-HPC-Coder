# 🚀 DeepSeek-Coder Fine-tuning Project

Projeto de fine-tuning para modelos DeepSeek-Coder usando LoRA e múltiplos datasets de código.

## 📚 Documentação

- **[QUICKSTART.md](QUICKSTART.md)** - Guia de início rápido
- **[DATASET_LOADER_GUIDE.md](DATASET_LOADER_GUIDE.md)** - Guia do módulo de datasets

## ⚡ Quick Start

### 1. Instalar Dependências
```bash
# Instalar dependências
pip install -r requirements.txt
```

### 2. Treinar Modelo (Comando Principal)
```bash
python3 finetune_with_args.py \
  --model_name deepseek-ai/deepseek-coder-1.3b-instruct \
  --output_dir ./models/gppd-hpc-cuda-coder-instruct \
  --save_dir ./checkpoints/gppd-hpc-cuda-coder-instruct \
  --use_evol_instruct \
  --use_magicoder \
  --log_file ./logs/gppd-hpc-cuda-coder-instruct.log
```

## 🛠️ Script Principal

**`finetune_with_args.py`** - Script configurável para fine-tuning com suporte a múltiplos datasets

### Recursos
- ✅ Suporte a múltiplos datasets (HPC-Instruct, Evol-Instruct, Magicoder)
- ✅ LoRA (Low-Rank Adaptation) para fine-tuning eficiente
- ✅ Quantização 8-bit para economia de memória
- ✅ Logging detalhado com timestamps
- ✅ Salvamento automático de adaptadores e modelo mesclado

## 📖 Exemplos de Uso

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

### Monitorar GPU Durante Treinamento (se disponível)
```bash
python monitor_gpu.py
```

## 🎯 Parâmetros Principais

| Parâmetro | Descrição | Padrão |
|-----------|-----------|---------|
| `--model_name` | Nome/caminho do modelo base | **obrigatório** |
| `--output_dir` | Diretório para checkpoints | `./results` |
| `--save_dir` | Diretório do modelo final | `./trained_model` |
| `--use_evol_instruct` | Adicionar dataset Evol-Instruct | `False` |
| `--use_magicoder` | Adicionar dataset Magicoder | `False` |
| `--log_file` | Arquivo de log | `None` |
| `--epochs` | Número de épocas | `3` |
| `--max_length` | Comprimento máximo de tokens | `512` |

## Estrutura de Saída

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

## 🔗 Links Úteis

- [Guia Rápido](QUICKSTART.md)
- [Guia Dataset Loader](DATASET_LOADER_GUIDE.md)
- [DeepSeek-Coder GitHub](https://github.com/deepseek-ai/DeepSeek-Coder)

## 📝 Licença

Este projeto é parte de um TCC (Trabalho de Conclusão de Curso).
