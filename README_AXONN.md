# Fine-tuning Distribuído com AxoNN

Este documento fornece um guia rápido para executar fine-tuning distribuído usando AxoNN no cluster HPC.

## 🚀 Quick Start

### 1. Instalar Dependências

```bash
# Criar e ativar ambiente virtual
python3 -m venv venv_axonn
source venv_axonn/bin/activate

# Instalar PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar dependências
pip install -r requirements.txt
```

### 2. Alocar Nós

```bash
# Alocar 2 nós com 4 GPUs cada (8 GPUs total)
salloc -p tupi -N 2 --gres=gpu:4 -J finetune-axonn -t 10:00:00
```

### 3. Executar Treinamento

#### Modo Interativo

```bash
source venv_axonn/bin/activate

mpirun -np 8 --map-by ppr:4:node python finetune_with_axonn.py \
    --model_name deepseek-ai/deepseek-coder-6.7b-base \
    --save_dir ./models/my-finetuned-model \
    --tensor_parallelism 2 \
    --data_parallelism 4 \
    --epochs 3
```

#### Modo Batch

```bash
# Editar train_axonn.sh conforme necessário
sbatch train_axonn.sh

# Monitorar
squeue -u $USER
tail -f logs/slurm-<JOB_ID>.out
```

## 📋 Parâmetros Principais

### Obrigatórios

- `--model_name`: Modelo do HuggingFace (ex: `deepseek-ai/deepseek-coder-6.7b-base`)
- `--save_dir`: Diretório onde salvar o modelo treinado

### Paralelização

- `--tensor_parallelism`: Divide o modelo entre GPUs (padrão: 1)
- `--data_parallelism`: Divide os dados entre GPUs (padrão: 1)

**Regra:** `total_gpus = tensor_parallelism × data_parallelism`

### Dataset

- `--dataset_name`: Dataset principal (padrão: `hpcgroup/hpc-instruct`)
- `--filter_language`: Filtrar por linguagem (padrão: `Cuda`)
- `--use_evol_instruct`: Adicionar dataset Evol-Instruct
- `--use_magicoder`: Adicionar dataset Magicoder

### Treinamento

- `--epochs`: Número de épocas (padrão: 3)
- `--per_device_train_batch_size`: Batch size por GPU (padrão: 2)
- `--gradient_accumulation_steps`: Acumulação de gradiente (padrão: 4)
- `--learning_rate`: Taxa de aprendizado (padrão: 2e-4)

### LoRA

- `--lora_r`: Rank do LoRA (padrão: 16)
- `--lora_alpha`: Alpha do LoRA (padrão: 32)
- `--lora_dropout`: Dropout do LoRA (padrão: 0.05)

## 📊 Configurações Recomendadas

### Modelo 1.3B - 1 Nó (4 GPUs)

```bash
mpirun -np 4 python finetune_with_axonn.py \
    --model_name deepseek-ai/deepseek-coder-1.3b-base \
    --save_dir ./models/deepseek-1.3b-finetuned \
    --tensor_parallelism 1 \
    --data_parallelism 4 \
    --per_device_train_batch_size 4
```

### Modelo 6.7B - 2 Nós (8 GPUs)

```bash
mpirun -np 8 --map-by ppr:4:node python finetune_with_axonn.py \
    --model_name deepseek-ai/deepseek-coder-6.7b-base \
    --save_dir ./models/deepseek-6.7b-finetuned \
    --tensor_parallelism 2 \
    --data_parallelism 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4
```

### Modelo 13B+ - 4 Nós (16 GPUs)

```bash
mpirun -np 16 --map-by ppr:4:node python finetune_with_axonn.py \
    --model_name deepseek-ai/deepseek-coder-13b-base \
    --save_dir ./models/deepseek-13b-finetuned \
    --tensor_parallelism 4 \
    --data_parallelism 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8
```

## 🔍 Monitoramento

### Ver Status do Job

```bash
squeue -u $USER
```

### Ver Logs em Tempo Real

```bash
# Log do SLURM
tail -f logs/slurm-<JOB_ID>.out

# Log do treinamento
tail -f logs/training_<JOB_ID>.log
```

### Monitorar GPUs

```bash
watch -n 1 srun nvidia-smi
```

## 📁 Estrutura de Saída

Após o treinamento, o modelo será salvo em:

```
models/my-finetuned-model/
├── lora_adapters/          # Adaptadores LoRA (leve, ~100MB)
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files...
└── merged_model/           # Modelo completo mesclado (pesado, ~13GB para 6.7B)
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files...
```

## ⚙️ Comandos Úteis

### Verificar Nós Disponíveis

```bash
sinfo -p tupi
sinfo -p tupi -t idle
```

### Cancelar Job

```bash
# Modo interativo
exit

# Modo batch
scancel <JOB_ID>
```

### Liberar Alocação

```bash
exit  # ou Ctrl+D
```

## 🐛 Troubleshooting

### Out of Memory (OOM)

Reduza o batch size:
```bash
--per_device_train_batch_size 1
--gradient_accumulation_steps 8
```

### NCCL Error

Adicione variáveis de ambiente:
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
```

### MPI Not Found

```bash
module load openmpi/4.1.0
pip install mpi4py --no-cache-dir
```

## 📚 Documentação Completa

Para mais detalhes, consulte:
- **[AXONN_TRAINING_GUIDE.md](AXONN_TRAINING_GUIDE.md)** - Guia completo com todos os comandos
- **[comparacao_frameworks.md](docs/comparacao_frameworks.md)** - Comparação SFTTrainer vs AxoNN
- **[DATASET_LOADER_GUIDE.md](DATASET_LOADER_GUIDE.md)** - Como usar diferentes datasets

## 🆚 AxoNN vs SFTTrainer

| Aspecto | SFTTrainer | AxoNN |
|---------|-----------|-------|
| **Facilidade** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Escalabilidade** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Performance** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Uso recomendado** | Modelos < 7B | Modelos > 7B |

**Use AxoNN quando:**
- Modelo > 7B parâmetros
- Múltiplos nós disponíveis
- Necessita máxima performance
- Ambiente HPC/cluster

**Use SFTTrainer quando:**
- Modelo < 7B parâmetros
- Single GPU ou poucos GPUs
- Prototipagem rápida
- Primeira vez fazendo fine-tuning

## 📞 Suporte

Para problemas:
1. Verifique os logs em `logs/`
2. Consulte [AXONN_TRAINING_GUIDE.md](AXONN_TRAINING_GUIDE.md)
3. Documentação AxoNN: https://axonn.readthedocs.io
