# Fine-tuning DeepSeek-Coder-6.7B em RTX 4090

Implementação otimizada para realizar fine-tuning do modelo DeepSeek-Coder-6.7B em uma GPU RTX 4090 (24GB VRAM) utilizando técnicas avançadas de otimização de memória.

## 🎯 Otimizações Implementadas

### 1. **QLoRA (4-bit Quantization)**
- Quantização 4-bit com double quantization (NF4)
- Reduz uso de memória em ~75% comparado com FP16
- Mantém qualidade do treinamento próxima ao full precision

### 2. **Gradient Checkpointing**
- Recomputa ativações durante backpropagation
- Reduz memória de ativações significativamente
- Trade-off: ~20% mais lento, mas viabiliza o treinamento

### 3. **LoRA (Low-Rank Adaptation)**
- Apenas 0.5-2% dos parâmetros são treináveis
- Rank configurável (8, 16, 32, 64)
- Treina matrizes de baixo rank ao invés do modelo completo

### 4. **Flash Attention 2 (Opcional)**
- Atenção otimizada O(N) ao invés de O(N²)
- Reduz uso de memória e aumenta velocidade
- Requer instalação separada

### 5. **Paged Optimizers**
- Usa `paged_adamw_32bit` para estados do otimizador
- Gerencia memória de forma mais eficiente
- Evita OOM em modelos grandes

### 6. **Gradient Accumulation**
- Simula batches maiores sem aumentar memória
- Batch efetivo = batch_size × accumulation_steps
- Exemplo: 1 × 16 = batch efetivo de 16

### 7. **Mixed Precision Training**
- BF16 (preferido) ou FP16
- Reduz uso de memória pela metade
- Acelera computação em GPUs modernas

## 📋 Requisitos

### Hardware
- GPU: RTX 4090 (24GB VRAM)
- RAM: Mínimo 32GB recomendado
- Storage: ~50GB livre (modelo + checkpoints)

### Software
```bash
pip install -r requirements.txt
```

**Flash Attention 2 (Opcional mas Recomendado):**
```bash
pip install flash-attn --no-build-isolation
```

## 🚀 Uso Básico

### Treinamento Padrão
```bash
python finetune_deepseek_optimized.py \
  --model_name deepseek-ai/deepseek-coder-6.7b-base \
  --output_dir ./results_deepseek \
  --save_dir ./trained_deepseek \
  --epochs 3 \
  --max_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16
```

### Com Flash Attention 2
```bash
python finetune_deepseek_optimized.py \
  --model_name deepseek-ai/deepseek-coder-6.7b-base \
  --use_flash_attention \
  --packing \
  --max_length 2048
```

### Teste Rápido (100 amostras)
```bash
python finetune_deepseek_optimized.py \
  --model_name deepseek-ai/deepseek-coder-6.7b-base \
  --max_samples 100 \
  --epochs 1 \
  --save_steps 50
```

## ⚙️ Parâmetros Principais

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--model_name` | `deepseek-ai/deepseek-coder-6.7b-base` | Modelo base |
| `--max_length` | `1024` | Comprimento máximo de sequência |
| `--per_device_train_batch_size` | `1` | Batch size (manter em 1) |
| `--gradient_accumulation_steps` | `16` | Acumulação de gradiente |
| `--lora_r` | `64` | Rank do LoRA |
| `--lora_alpha` | `128` | Alpha do LoRA (2× rank) |
| `--learning_rate` | `2e-4` | Taxa de aprendizado |
| `--epochs` | `3` | Número de épocas |
| `--use_flash_attention` | `False` | Habilitar Flash Attention 2 |
| `--packing` | `False` | Empacotar sequências |

## 📊 Monitoramento de Memória

### Monitor em Tempo Real
```bash
python monitor_gpu.py
```

### Ver Apenas GPU
```bash
python monitor_gpu.py --no-cpu
```

### Snapshot Único
```bash
python monitor_gpu.py --once
```

## 🎛️ Ajuste Fino de Hiperparâmetros

### Reduzir Uso de Memória

1. **Diminuir max_length:**
```bash
--max_length 512
```

2. **LoRA com rank menor:**
```bash
--lora_r 32 --lora_alpha 64
```

3. **Aumentar gradient accumulation:**
```bash
--gradient_accumulation_steps 32
```

### Aumentar Qualidade do Treinamento

1. **LoRA com rank maior:**
```bash
--lora_r 128 --lora_alpha 256
```

2. **Sequências mais longas:**
```bash
--max_length 2048 --use_flash_attention
```

3. **Batch efetivo maior:**
```bash
--gradient_accumulation_steps 32
```

## 💾 Uso de Memória Estimado

| Configuração | Memória GPU | Batch Efetivo | Velocidade |
|--------------|-------------|---------------|------------|
| Conservative | ~18GB | 8 | Rápido |
| Balanced | ~20GB | 16 | Médio |
| Quality | ~22GB | 32 | Lento |

**Conservative:**
```bash
--max_length 512 --lora_r 32 --gradient_accumulation_steps 8
```

**Balanced (Recomendado):**
```bash
--max_length 1024 --lora_r 64 --gradient_accumulation_steps 16
```

**Quality:**
```bash
--max_length 2048 --lora_r 128 --gradient_accumulation_steps 32 --use_flash_attention
```

## 🔧 Inferência com Modelo Treinado

### Usando Adaptadores LoRA
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-base",
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "./trained_deepseek")
tokenizer = AutoTokenizer.from_pretrained("./trained_deepseek")

prompt = "### Instruction:\nWrite a CUDA kernel for matrix multiplication\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

### Usando Modelo Mesclado
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./trained_deepseek/merged_model",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./trained_deepseek/merged_model")
```

## 🐛 Troubleshooting

### OOM (Out of Memory)
1. Reduzir `max_length` para 512 ou 256
2. Reduzir `lora_r` para 32 ou 16
3. Aumentar `gradient_accumulation_steps`
4. Fechar outros programas usando GPU

### Treinamento Muito Lento
1. Habilitar `--use_flash_attention`
2. Habilitar `--packing`
3. Aumentar `per_device_train_batch_size` se houver memória
4. Reduzir `dataloader_num_workers`

### Erro de Flash Attention
```bash
pip uninstall flash-attn
pip install flash-attn --no-build-isolation
```
Ou remover `--use_flash_attention` da linha de comando.

### CUDA Out of Memory Durante Salvamento
```python
model.config.use_cache = False
```
Já implementado no script. Se persistir, salvar apenas adaptadores LoRA.

## 📈 Benchmark de Performance

Baseado em testes com RTX 4090:

| Config | Tokens/s | Memória | Tempo/Epoch |
|--------|----------|---------|-------------|
| Conservative | ~1200 | 18GB | 45min |
| Balanced | ~1000 | 20GB | 55min |
| Quality | ~800 | 22GB | 70min |

*Valores aproximados, variam com dataset e hardware*

## 📚 Estrutura de Arquivos

```
trained_deepseek/
├── adapter_config.json      # Configuração LoRA
├── adapter_model.bin         # Pesos dos adaptadores
├── tokenizer_config.json    # Config do tokenizer
├── tokenizer.json           # Tokenizer
└── merged_model/            # Modelo completo mesclado
    ├── model.safetensors.index.json
    ├── model-00001-of-00003.safetensors
    ├── model-00002-of-00003.safetensors
    └── model-00003-of-00003.safetensors
```

## 🎓 Referências

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Flash Attention 2](https://arxiv.org/abs/2307.08691)
- [DeepSeek-Coder](https://github.com/deepseek-ai/DeepSeek-Coder)

## 📝 Notas

- O script usa automaticamente BF16 se a GPU suportar
- Checkpoints são salvos a cada 250 steps
- Apenas os 2 últimos checkpoints são mantidos para economizar espaço
- O modelo mesclado é opcional mas facilita deployment
