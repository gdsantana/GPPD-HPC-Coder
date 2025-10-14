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

echo "=========================================="
echo "Iniciando treinamento distribuído com AxoNN"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nós alocados: $SLURM_NODELIST"
echo "Número de nós: $SLURM_NNODES"
echo "GPUs por nó: 4"
echo "Total de GPUs: $(($SLURM_NNODES * 4))"
echo "=========================================="

# Criar diretório de logs
mkdir -p logs
mkdir -p models

# Carregar módulos necessários
echo "Carregando módulos..."
module load python/3.10
module load cuda/11.8
module load openmpi/4.1.0

# Ativar ambiente virtual
echo "Ativando ambiente virtual..."
source venv_axonn/bin/activate

# Configurar variáveis de ambiente para NCCL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# Configurações do modelo e treinamento
MODEL_NAME="deepseek-ai/deepseek-coder-6.7b-base"
SAVE_DIR="./models/deepseek-6.7b-cuda-finetuned-$(date +%Y%m%d_%H%M%S)"
TENSOR_PARALLEL=2
DATA_PARALLEL=4
EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=4
LEARNING_RATE=2e-4

echo "=========================================="
echo "Configurações do treinamento:"
echo "  Modelo: $MODEL_NAME"
echo "  Diretório de salvamento: $SAVE_DIR"
echo "  Tensor Parallelism: $TENSOR_PARALLEL"
echo "  Data Parallelism: $DATA_PARALLEL"
echo "  Épocas: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Learning rate: $LEARNING_RATE"
echo "=========================================="

# Executar treinamento
echo "Iniciando treinamento..."
srun python finetune_with_axonn.py \
    --model_name "$MODEL_NAME" \
    --save_dir "$SAVE_DIR" \
    --tensor_parallelism $TENSOR_PARALLEL \
    --data_parallelism $DATA_PARALLEL \
    --epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --filter_language Cuda \
    --log_file "logs/training_${SLURM_JOB_ID}.log"

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Treinamento concluído com sucesso!"
    echo "Modelo salvo em: $SAVE_DIR"
else
    echo "Treinamento falhou com código de saída: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
