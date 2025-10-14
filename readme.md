# 🚀 DeepSeek-Coder Fine-tuning Project

Projeto completo de fine-tuning para modelos DeepSeek-Coder com otimizações avançadas para RTX 4090.

## 📚 Documentação

- **[README_DEEPSEEK_FINETUNE.md](README_DEEPSEEK_FINETUNE.md)** - Documentação completa do projeto
- **[QUICKSTART.md](QUICKSTART.md)** - Guia de início rápido
- **[DATASET_LOADER_GUIDE.md](DATASET_LOADER_GUIDE.md)** - 🆕 Guia do módulo de datasets
- **[REFACTORING_NOTES.md](REFACTORING_NOTES.md)** - 🆕 Notas sobre refatoração

## ⚡ Quick Start

### 1. Instalar Dependências
```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 2. Verificar Compatibilidade
```bash
python check_compatibility.py
```

### 3. Treinar Modelo (Configuração Recomendada)
```bash
python finetune_deepseek_optimized.py \
  --model_name deepseek-ai/deepseek-coder-6.7b-base \
  --output_dir ./results \
  --save_dir ./trained_model \
  --epochs 3 \
  --max_length 1024 \
  --lora_r 64 \
  --gradient_accumulation_steps 16 \
  --use_flash_attention \
  --packing
```

### 4. Inferência
```bash
python inference_example.py \
  --adapter_path ./trained_model \
  --interactive
```

## Estrutura de Diretórios

Cada modelo fine-tuned é salvo com a seguinte estrutura:

```
finetune_model-{n_params}/
├── lora_adapters/          # Adaptadores LoRA (PEFT)
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
├── merged_model/           # Modelo completo mesclado (opcional)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

## 🛠️ Scripts Principais

| Script | Descrição |
|--------|-----------|
| `finetune_deepseek_optimized.py` | Script principal com todas as otimizações |
| `finetune_with_args.py` | Script configurável via argumentos |
| `check_compatibility.py` | Verificador de compatibilidade do sistema |
| `monitor_gpu.py` | Monitor de memória GPU em tempo real |
| `inference_example.py` | Script de inferência com modelo treinado |
| `generate_prompts.py` | Gerador de prompts em lote |
| `dataset_loader.py` | Módulo modular de datasets |

## 📖 Exemplos de Uso

### Treinamento Básico
```bash
python finetune_deepseek_optimized.py
```

### Treinamento com Múltiplos Datasets
```bash
python finetune_deepseek_optimized.py \
  --dataset_names hpcgroup/hpc-instruct bigcode/the-stack \
  --dataset_configs default python
```

### Monitorar GPU Durante Treinamento
```bash
python monitor_gpu.py
```

### Gerar Prompts
```bash
python generate_prompts.py \
  --model_dir ./trained_model \
  --prompts_dir ./prompts \
  --output_dir ./outputs \
  --k 5
```

## 🎯 Configurações Recomendadas

### Conservative (~18GB VRAM)
```bash
--max_length 512 --lora_r 32 --gradient_accumulation_steps 8
```

### Balanced (~20GB VRAM) - Recomendado
```bash
--max_length 1024 --lora_r 64 --gradient_accumulation_steps 16
```

### Quality (~22GB VRAM)
```bash
--max_length 2048 --lora_r 128 --gradient_accumulation_steps 32 --use_flash_attention
```

## 🔗 Links Úteis

- [Documentação Completa](README_DEEPSEEK_FINETUNE.md)
- [Guia Rápido](QUICKSTART.md)
- [Guia Dataset Loader](DATASET_LOADER_GUIDE.md)
- [DeepSeek-Coder GitHub](https://github.com/deepseek-ai/DeepSeek-Coder)

## 📝 Licença

Este projeto é parte de um TCC (Trabalho de Conclusão de Curso).
