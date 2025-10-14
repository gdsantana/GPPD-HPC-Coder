# Guia de Treinamento Distribuído com AxoNN

Este guia explica como executar o fine-tuning distribuído usando AxoNN no cluster HPC.

## Índice
1. [Pré-requisitos](#pré-requisitos)
2. [Instalação](#instalação)
3. [Alocação de Nós](#alocação-de-nós)
4. [Execução do Treinamento](#execução-do-treinamento)
5. [Exemplos de Uso](#exemplos-de-uso)
6. [Troubleshooting](#troubleshooting)

---

## Pré-requisitos

### Módulos do Sistema
Certifique-se de que os seguintes módulos estão carregados:

```bash
module load python/3.10
module load cuda/11.8
module load openmpi/4.1.0
```

### Verificar Disponibilidade dos Nós

Antes de alocar, verifique quais nós estão disponíveis:

```bash
# Ver status da partição tupi
sinfo -p tupi

# Ver detalhes dos nós
scontrol show nodes tupi[1-6]

# Ver nós disponíveis
sinfo -p tupi -t idle
```

---

## Instalação

### 1. Criar Ambiente Virtual

```bash
cd /home/gdsantana/TCC
python3 -m venv venv_axonn
source venv_axonn/bin/activate
```

### 2. Instalar Dependências

```bash
# Atualizar pip
pip install --upgrade pip

# Instalar PyTorch (ajuste a versão CUDA conforme necessário)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar dependências do projeto
pip install -r requirements.txt
```

### 3. Verificar Instalação do AxoNN

```bash
python -c "import axonn; print('AxoNN instalado com sucesso!')"
python -c "from mpi4py import MPI; print('MPI4PY instalado com sucesso!')"
```

---

## Alocação de Nós

### Modo Interativo

#### Alocar 1 Nó (4 GPUs)
```bash
salloc -p tupi -N 1 --gres=gpu:4 -J finetune-axonn -t 10:00:00
```

#### Alocar 2 Nós (8 GPUs)
```bash
salloc -p tupi -N 2 --gres=gpu:4 -J finetune-axonn -t 10:00:00
```

#### Alocar 4 Nós (16 GPUs)
```bash
salloc -p tupi -N 4 --gres=gpu:4 -J finetune-axonn -t 10:00:00
```

#### Alocar Nós Específicos
```bash
# Alocar tupi6 e tupi7
salloc -p tupi -w tupi[6-7] --gres=gpu:4 -J finetune-axonn -t 10:00:00

# Alocar tupi6, tupi7, tupi8
salloc -p tupi -w tupi[6-8] --gres=gpu:4 -J finetune-axonn -t 10:00:00
```

### Verificar Alocação

Após a alocação, verifique os nós atribuídos:

```bash
echo $SLURM_NODELIST
echo $SLURM_JOB_NODELIST
squeue -u $USER
```

---

## Execução do Treinamento

### Modo Interativo (após salloc)

#### Exemplo 1: Treinamento com 2 Nós (8 GPUs)

```bash
# Ativar ambiente
source venv_axonn/bin/activate

# Executar com mpirun
mpirun -np 8 \
    --map-by ppr:4:node \
    python finetune_with_axonn.py \
    --model_name deepseek-ai/deepseek-coder-6.7b-base \
    --save_dir ./models/deepseek-6.7b-cuda-finetuned \
    --tensor_parallelism 2 \
    --data_parallelism 4 \
    --epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --filter_language Cuda \
    --log_file logs/training_axonn.log
```

#### Exemplo 2: Com Datasets Adicionais

```bash
mpirun -np 8 \
    --map-by ppr:4:node \
    python finetune_with_axonn.py \
    --model_name deepseek-ai/deepseek-coder-6.7b-base \
    --save_dir ./models/deepseek-6.7b-multi-dataset \
    --tensor_parallelism 2 \
    --data_parallelism 4 \
    --epochs 3 \
    --per_device_train_batch_size 2 \
    --use_evol_instruct \
    --use_magicoder \
    --log_file logs/training_multi_dataset.log
```

### Modo Batch (sbatch)

Crie um script de batch `train_axonn.sh`:

```bash
#!/bin/bash
#SBATCH -p tupi
#SBATCH -N 2                      # 2 nós
#SBATCH --ntasks-per-node=4       # 4 tarefas por nó (1 por GPU)
#SBATCH --gres=gpu:4              # 4 GPUs por nó
#SBATCH --cpus-per-task=8         # CPUs por tarefa
#SBATCH -J finetune-axonn
#SBATCH -t 10:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

# Criar diretório de logs
mkdir -p logs

# Carregar módulos
module load python/3.10
module load cuda/11.8
module load openmpi/4.1.0

# Ativar ambiente virtual
source venv_axonn/bin/activate

# Configurar variáveis de ambiente
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# Executar treinamento
srun python finetune_with_axonn.py \
    --model_name deepseek-ai/deepseek-coder-6.7b-base \
    --save_dir ./models/deepseek-6.7b-cuda-finetuned \
    --tensor_parallelism 2 \
    --data_parallelism 4 \
    --epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --filter_language Cuda \
    --log_file logs/training_${SLURM_JOB_ID}.log

echo "Treinamento concluído!"
```

Submeter o job:

```bash
sbatch train_axonn.sh
```

Monitorar o job:

```bash
# Ver status do job
squeue -u $USER

# Ver saída em tempo real
tail -f logs/slurm-<JOB_ID>.out

# Ver log de treinamento
tail -f logs/training_<JOB_ID>.log
```

---

## Exemplos de Uso

### 1. Modelo Pequeno (1.3B) - 1 Nó

```bash
salloc -p tupi -N 1 --gres=gpu:4 -J finetune-1.3b -t 5:00:00

mpirun -np 4 python finetune_with_axonn.py \
    --model_name deepseek-ai/deepseek-coder-1.3b-base \
    --save_dir ./models/deepseek-1.3b-finetuned \
    --tensor_parallelism 1 \
    --data_parallelism 4 \
    --epochs 3 \
    --per_device_train_batch_size 4
```

### 2. Modelo Médio (6.7B) - 2 Nós

```bash
salloc -p tupi -N 2 --gres=gpu:4 -J finetune-6.7b -t 10:00:00

mpirun -np 8 --map-by ppr:4:node python finetune_with_axonn.py \
    --model_name deepseek-ai/deepseek-coder-6.7b-base \
    --save_dir ./models/deepseek-6.7b-finetuned \
    --tensor_parallelism 2 \
    --data_parallelism 4 \
    --epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4
```

### 3. Modelo Grande (13B+) - 4 Nós

```bash
salloc -p tupi -N 4 --gres=gpu:4 -J finetune-13b -t 20:00:00

mpirun -np 16 --map-by ppr:4:node python finetune_with_axonn.py \
    --model_name deepseek-ai/deepseek-coder-13b-base \
    --save_dir ./models/deepseek-13b-finetuned \
    --tensor_parallelism 4 \
    --data_parallelism 4 \
    --epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8
```

### 4. Treinamento com Todos os Datasets

```bash
mpirun -np 8 --map-by ppr:4:node python finetune_with_axonn.py \
    --model_name deepseek-ai/deepseek-coder-6.7b-base \
    --save_dir ./models/deepseek-6.7b-all-datasets \
    --tensor_parallelism 2 \
    --data_parallelism 4 \
    --epochs 3 \
    --per_device_train_batch_size 2 \
    --use_evol_instruct \
    --use_magicoder \
    --filter_language Cuda
```

---

## Parâmetros Importantes

### Paralelização

- **`--tensor_parallelism`**: Divide o modelo entre GPUs (use para modelos grandes)
  - Valor recomendado: 1-4 dependendo do tamanho do modelo
  
- **`--data_parallelism`**: Divide os dados entre GPUs (use para acelerar treinamento)
  - Valor recomendado: número de GPUs / tensor_parallelism

### Regra de Ouro

```
total_gpus = tensor_parallelism × data_parallelism
```

**Exemplos:**
- 8 GPUs: `--tensor_parallelism 2 --data_parallelism 4`
- 16 GPUs: `--tensor_parallelism 4 --data_parallelism 4`
- 4 GPUs: `--tensor_parallelism 1 --data_parallelism 4`

### Batch Size e Memória

Para evitar OOM (Out of Memory):

```bash
# Modelo 6.7B - 2 GPUs por tensor parallel
--per_device_train_batch_size 2
--gradient_accumulation_steps 4

# Modelo 13B - 4 GPUs por tensor parallel
--per_device_train_batch_size 1
--gradient_accumulation_steps 8
```

---

## Troubleshooting

### Erro: "NCCL initialization failed"

```bash
# Adicionar antes do mpirun:
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
```

### Erro: "Out of Memory"

Reduza o batch size ou aumente gradient accumulation:

```bash
--per_device_train_batch_size 1
--gradient_accumulation_steps 8
```

### Erro: "MPI not found"

```bash
module load openmpi/4.1.0
pip install mpi4py --no-cache-dir
```

### Verificar GPUs Disponíveis

```bash
# Em cada nó alocado
srun --nodes=1 --ntasks=1 nvidia-smi

# Ver todas as GPUs
srun nvidia-smi
```

### Monitorar Uso de GPU Durante Treinamento

```bash
# Terminal 1: Executar treinamento
mpirun -np 8 python finetune_with_axonn.py ...

# Terminal 2: Monitorar GPUs
watch -n 1 srun nvidia-smi
```

### Cancelar Job

```bash
# Modo interativo
exit

# Modo batch
scancel <JOB_ID>

# Cancelar todos os seus jobs
scancel -u $USER
```

---

## Estrutura de Diretórios Após Treinamento

```
models/deepseek-6.7b-cuda-finetuned/
├── lora_adapters/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files...
└── merged_model/
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files...
```

---

## Comparação de Performance

| Configuração | Nós | GPUs | Tempo Estimado (3 épocas) |
|--------------|-----|------|---------------------------|
| 1.3B - 1 nó  | 1   | 4    | ~2-3 horas                |
| 6.7B - 2 nós | 2   | 8    | ~4-6 horas                |
| 13B - 4 nós  | 4   | 16   | ~8-12 horas               |

*Tempos estimados para dataset hpc-instruct filtrado por Cuda (~1000 exemplos)*

---

## Próximos Passos

Após o treinamento, veja:
- `README_DEEPSEEK_FINETUNE.md` - Como usar o modelo treinado
- `DATASET_LOADER_GUIDE.md` - Como adicionar novos datasets
- `comparacao_frameworks.md` - Comparação entre SFTTrainer e AxoNN

---

## Suporte

Para problemas ou dúvidas:
1. Verifique os logs em `logs/`
2. Consulte a documentação do AxoNN: https://axonn.readthedocs.io
3. Verifique o status do cluster: `sinfo -p tupi`
