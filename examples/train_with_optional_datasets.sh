#!/bin/bash
# Exemplos de uso dos datasets opcionais

echo "=========================================="
echo "Exemplos de Fine-tuning com Datasets Opcionais"
echo "=========================================="
echo ""

# Exemplo 1: Apenas dataset padrão (HPC-Instruct)
echo "1. Dataset padrão apenas (HPC-Instruct com filtro CUDA)"
echo "   python finetune_deepseek_optimized.py \\"
echo "       --model_name deepseek-ai/deepseek-coder-6.7b-base \\"
echo "       --output_dir ./results_hpc_only \\"
echo "       --epochs 3"
echo ""

# Exemplo 2: HPC-Instruct + Evol-Instruct
echo "2. HPC-Instruct + Evol-Instruct-Code-80k-v1"
echo "   python finetune_deepseek_optimized.py \\"
echo "       --model_name deepseek-ai/deepseek-coder-6.7b-base \\"
echo "       --output_dir ./results_with_evol \\"
echo "       --epochs 3 \\"
echo "       --use_evol_instruct"
echo ""

# Exemplo 3: HPC-Instruct + Magicoder
echo "3. HPC-Instruct + Magicoder-OSS-Instruct-75K"
echo "   python finetune_deepseek_optimized.py \\"
echo "       --model_name deepseek-ai/deepseek-coder-6.7b-base \\"
echo "       --output_dir ./results_with_magicoder \\"
echo "       --epochs 3 \\"
echo "       --use_magicoder"
echo ""

# Exemplo 4: Todos os datasets
echo "4. Todos os datasets (HPC + Evol-Instruct + Magicoder)"
echo "   python finetune_deepseek_optimized.py \\"
echo "       --model_name deepseek-ai/deepseek-coder-6.7b-base \\"
echo "       --output_dir ./results_all_datasets \\"
echo "       --epochs 3 \\"
echo "       --use_evol_instruct \\"
echo "       --use_magicoder"
echo ""

# Exemplo 5: Teste rápido com 1 época
echo "5. Teste rápido com 1 época"
echo "   python finetune_deepseek_optimized.py \\"
echo "       --model_name deepseek-ai/deepseek-coder-6.7b-base \\"
echo "       --output_dir ./results_quick_test \\"
echo "       --epochs 1 \\"
echo "       --use_evol_instruct \\"
echo "       --use_magicoder"
echo ""

# Exemplo 6: Usando com finetune_with_args.py
echo "6. Usando finetune_with_args.py com datasets opcionais"
echo "   python finetune_with_args.py \\"
echo "       --model_name deepseek-ai/deepseek-coder-1.3b-base \\"
echo "       --output_dir ./results_1.3b \\"
echo "       --epochs 3 \\"
echo "       --use_evol_instruct \\"
echo "       --use_magicoder"
echo ""

# Exemplo 7: Usando com finetune_with_args_QLORA.py
echo "7. Usando finetune_with_args_QLORA.py com datasets opcionais"
echo "   python finetune_with_args_QLORA.py \\"
echo "       --model_name deepseek-ai/deepseek-coder-1.3b-base \\"
echo "       --output_dir ./results_qlora \\"
echo "       --epochs 3 \\"
echo "       --use_evol_instruct \\"
echo "       --use_magicoder"
echo ""

echo "=========================================="
echo "Para executar um exemplo, copie o comando e execute no terminal"
echo "=========================================="
