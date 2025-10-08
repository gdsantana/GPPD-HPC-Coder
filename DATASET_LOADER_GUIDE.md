# 📚 Guia do Módulo Dataset Loader

## 🎯 Visão Geral

O módulo `dataset_loader.py` fornece uma interface modular e reutilizável para carregar e preparar datasets para fine-tuning de modelos de linguagem. Ele suporta:

- ✅ **Dataset único ou múltiplos datasets**
- ✅ **Configurações personalizadas por dataset**
- ✅ **Colunas flexíveis** (instrução e resposta)
- ✅ **Filtragem por linguagem**
- ✅ **Limitação de amostras**
- ✅ **Templates de formatação customizáveis**
- ✅ **Fallbacks automáticos** para diferentes formatos de dataset
- ✅ **Combinação automática** de múltiplos datasets

## 📦 Componentes Principais

### 1. `DatasetConfig`

Classe de configuração para um dataset individual.

```python
class DatasetConfig:
    def __init__(
        self,
        name: str,                          # Nome do dataset no HuggingFace
        config: Optional[str] = None,       # Configuração específica do dataset
        instruction_column: str = "problem statement",  # Coluna de instrução
        response_column: str = "solution",  # Coluna de resposta
        filter_language: Optional[str] = None,  # Filtro de linguagem
        max_samples: Optional[int] = None   # Limite de amostras
    ):
```

**Parâmetros:**
- `name`: Nome do dataset no HuggingFace Hub (ex: `"hpcgroup/hpc-instruct"`)
- `config`: Configuração específica do dataset (ex: `"python"`, `"default"`)
- `instruction_column`: Nome da coluna que contém as instruções
- `response_column`: Nome da coluna que contém as respostas
- `filter_language`: Filtrar exemplos por linguagem (ex: `"Cuda"`)
- `max_samples`: Número máximo de amostras a carregar

### 2. `DatasetLoader`

Classe principal para carregar e processar datasets.

```python
class DatasetLoader:
    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: Tokenizer do modelo (ex: AutoTokenizer)
        """
```

**Métodos:**

#### `load_single_dataset()`
Carrega e processa um único dataset.

```python
def load_single_dataset(
    self,
    dataset_config: DatasetConfig,
    format_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
) -> Optional[Dataset]:
```

#### `load_multiple_datasets()`
Carrega e combina múltiplos datasets.

```python
def load_multiple_datasets(
    self,
    dataset_configs: List[DatasetConfig],
    format_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
) -> Dataset:
```

### 3. `load_dataset_from_args()`

Função de conveniência para carregar datasets a partir de argumentos CLI.

```python
def load_dataset_from_args(args, tokenizer) -> Dataset:
    """
    Carrega dataset(s) automaticamente baseado nos argumentos.
    Suporta tanto dataset único quanto múltiplos datasets.
    """
```

## 🚀 Exemplos de Uso

### Exemplo 1: Dataset Único Básico

```python
from transformers import AutoTokenizer
from dataset_loader import DatasetConfig, DatasetLoader

# Carregar tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base")

# Criar loader
loader = DatasetLoader(tokenizer)

# Configurar dataset
config = DatasetConfig(
    name="hpcgroup/hpc-instruct",
    filter_language="Cuda"
)

# Carregar dataset
dataset = loader.load_single_dataset(config)
print(f"Dataset carregado: {len(dataset)} exemplos")
```

### Exemplo 2: Dataset com Colunas Personalizadas

```python
from dataset_loader import DatasetConfig, DatasetLoader

loader = DatasetLoader(tokenizer)

config = DatasetConfig(
    name="tatsu-lab/alpaca",
    instruction_column="instruction",  # Colunas diferentes
    response_column="output",
    max_samples=1000  # Limitar a 1000 exemplos
)

dataset = loader.load_single_dataset(config)
```

### Exemplo 3: Template de Formatação Customizado

```python
# Template sem marcadores ###
custom_template = "Instruction: {instruction}\nResponse: {response}"

dataset = loader.load_single_dataset(config, format_template=custom_template)
```

### Exemplo 4: Múltiplos Datasets

```python
from dataset_loader import DatasetConfig, DatasetLoader

loader = DatasetLoader(tokenizer)

# Configurar múltiplos datasets
configs = [
    DatasetConfig(
        name="hpcgroup/hpc-instruct",
        filter_language="Cuda",
        max_samples=5000
    ),
    DatasetConfig(
        name="bigcode/the-stack",
        config="python",
        instruction_column="content",
        response_column="content",
        max_samples=3000
    ),
    DatasetConfig(
        name="tatsu-lab/alpaca",
        instruction_column="instruction",
        response_column="output",
        max_samples=2000
    )
]

# Carregar e combinar todos os datasets
combined_dataset = loader.load_multiple_datasets(configs)
print(f"Dataset combinado: {len(combined_dataset)} exemplos")
```

### Exemplo 5: Integração com Scripts de Treinamento

```python
#!/usr/bin/env python3
import argparse
from transformers import AutoTokenizer
from dataset_loader import load_dataset_from_args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="hpcgroup/hpc-instruct")
    parser.add_argument("--filter_language", type=str, default="Cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Carregar tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base")
    
    # Carregar dataset automaticamente dos argumentos
    dataset = load_dataset_from_args(args, tokenizer)
    
    print(f"Dataset pronto: {len(dataset)} exemplos")
    # Continuar com o treinamento...

if __name__ == "__main__":
    main()
```

## 🔧 Argumentos CLI Suportados

O módulo reconhece automaticamente os seguintes argumentos quando usado com `load_dataset_from_args()`:

### Dataset Único:
- `--dataset_name`: Nome do dataset
- `--dataset_config`: Configuração específica
- `--filter_language`: Filtro de linguagem
- `--max_samples`: Limite de amostras
- `--dataset_columns`: Colunas no formato `"instrução,resposta"`

### Múltiplos Datasets:
- `--dataset_names`: Lista de nomes (ex: `dataset1 dataset2 dataset3`)
- `--dataset_configs`: Lista de configurações (mesma ordem)
- `--filter_language`: Aplicado a todos os datasets
- `--max_samples`: Dividido igualmente entre datasets
- `--dataset_columns`: Aplicado a todos os datasets

## 📊 Fallbacks Automáticos

O módulo tenta encontrar colunas automaticamente na seguinte ordem:

**Para Instrução:**
1. Coluna especificada em `instruction_column`
2. `"instruction"`
3. `"prompt"`
4. `"input"`
5. String vazia se nenhuma encontrada

**Para Resposta:**
1. Coluna especificada em `response_column`
2. `"response"`
3. `"output"`
4. `"completion"`
5. `"solution"`
6. String vazia se nenhuma encontrada

## 🎨 Templates de Formatação

### Template Padrão (DeepSeek Style)
```python
"### Instruction:\n{instruction}\n\n### Response:\n{response}"
```

**Resultado:**
```
### Instruction:
Write a CUDA kernel for vector addition

### Response:
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

### Template Simples
```python
"Instruction: {instruction}\nResponse: {response}"
```

**Resultado:**
```
Instruction: Write a CUDA kernel for vector addition
Response: __global__ void vectorAdd(...)
```

### Template Alpaca
```python
"Below is an instruction. Write a response.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"
```

### Template Personalizado
```python
"Q: {instruction}\nA: {response}"
```

## 🔍 Tratamento de Erros

O módulo possui tratamento robusto de erros:

```python
# Se um dataset falhar ao carregar, ele é pulado
dataset = loader.load_single_dataset(config)
if dataset is None:
    print("Falha ao carregar dataset")

# Para múltiplos datasets, pelo menos um deve ser carregado
try:
    combined = loader.load_multiple_datasets(configs)
except ValueError as e:
    print(f"Erro: {e}")  # "Nenhum dataset foi carregado com sucesso!"
```

## 📈 Logs e Monitoramento

O módulo fornece logs detalhados:

```
INFO: Carregando dataset: hpcgroup/hpc-instruct
INFO: Dataset 'hpcgroup/hpc-instruct' carregado
INFO: Total de exemplos: 15234
INFO: Filtro de linguagem 'Cuda': 15234 -> 8456 exemplos
INFO: Limitado a 5000 amostras
INFO: Formatando exemplos do dataset 'hpcgroup/hpc-instruct'...
INFO: ✓ Dataset 'hpcgroup/hpc-instruct' processado: 5000 exemplos
```

## 🔄 Migração de Código Existente

### Antes (Código Antigo):
```python
from datasets import load_dataset

dataset = load_dataset("hpcgroup/hpc-instruct")
cuda_dataset = dataset.filter(lambda x: x.get("language") == "Cuda")
train_dataset = cuda_dataset["train"]

def format_example(example):
    instruction = example.get('problem statement', '')
    response = example.get('solution', '')
    text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    return {"text": text}

formatted_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
```

### Depois (Com dataset_loader):
```python
from dataset_loader import DatasetConfig, DatasetLoader

loader = DatasetLoader(tokenizer)
config = DatasetConfig(
    name="hpcgroup/hpc-instruct",
    filter_language="Cuda"
)
formatted_dataset = loader.load_single_dataset(config)
```

## 🎯 Casos de Uso Comuns

### 1. Fine-tuning com Dataset HPC
```python
config = DatasetConfig(
    name="hpcgroup/hpc-instruct",
    filter_language="Cuda",
    max_samples=10000
)
dataset = loader.load_single_dataset(config)
```

### 2. Combinar Datasets de Código
```python
configs = [
    DatasetConfig(name="hpcgroup/hpc-instruct", filter_language="Cuda"),
    DatasetConfig(name="bigcode/the-stack", config="python"),
    DatasetConfig(name="codeparrot/github-code", config="cpp")
]
dataset = loader.load_multiple_datasets(configs)
```

### 3. Dataset de Instrução Geral
```python
config = DatasetConfig(
    name="tatsu-lab/alpaca",
    instruction_column="instruction",
    response_column="output",
    max_samples=50000
)
dataset = loader.load_single_dataset(config)
```

### 4. Dataset Personalizado
```python
config = DatasetConfig(
    name="seu-usuario/seu-dataset",
    instruction_column="question",
    response_column="answer"
)
custom_template = "Q: {instruction}\nA: {response}"
dataset = loader.load_single_dataset(config, format_template=custom_template)
```

## 🛠️ Integração com Scripts Existentes

### finetune_deepseek_optimized.py
```python
from dataset_loader import load_dataset_from_args

def load_and_prepare_dataset(args, tokenizer):
    return load_dataset_from_args(args, tokenizer)
```

### finetune_with_args.py
```python
from dataset_loader import DatasetConfig, DatasetLoader

loader = DatasetLoader(tokenizer)
config = DatasetConfig(
    name=args.dataset_name,
    instruction_column="problem statement",
    response_column="solution",
    filter_language=args.filter_language
)
format_template = "Instruction: {instruction}\nResponse: {response}"
dataset = loader.load_single_dataset(config, format_template)
```

## 📝 Notas Importantes

1. **Tokenizer**: O tokenizer deve ser passado ao criar o `DatasetLoader`
2. **EOS Token**: O token EOS é automaticamente adicionado ao final de cada exemplo
3. **Processamento Paralelo**: Usa `num_proc=4` por padrão para processamento mais rápido
4. **Memória**: Datasets grandes podem consumir muita memória - use `max_samples` para limitar
5. **Compatibilidade**: Funciona com qualquer dataset do HuggingFace Hub que tenha split "train"

## 🐛 Troubleshooting

### Erro: "Nenhum dataset foi carregado com sucesso!"
- Verifique se os nomes dos datasets estão corretos
- Confirme que você tem acesso aos datasets (alguns são privados)
- Verifique sua conexão com a internet

### Erro: Colunas não encontradas
- Use `dataset_columns` para especificar as colunas corretas
- Verifique a estrutura do dataset no HuggingFace Hub
- Os fallbacks automáticos tentarão encontrar colunas comuns

### Dataset vazio após filtragem
- Verifique se o filtro de linguagem está correto
- Confirme que o dataset tem exemplos na linguagem especificada
- Remova o filtro para ver todos os exemplos disponíveis

## 📚 Referências

- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
- [Transformers Tokenizers](https://huggingface.co/docs/transformers/main_classes/tokenizer)
- [Dataset Concatenation](https://huggingface.co/docs/datasets/process#concatenate)

## 🎓 Exemplos Avançados

### Balanceamento de Datasets
```python
# Carregar 5000 exemplos de cada dataset
configs = [
    DatasetConfig(name="dataset1", max_samples=5000),
    DatasetConfig(name="dataset2", max_samples=5000),
    DatasetConfig(name="dataset3", max_samples=5000)
]
balanced_dataset = loader.load_multiple_datasets(configs)
# Total: 15000 exemplos balanceados
```

### Pipeline Completo
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset_loader import DatasetConfig, DatasetLoader
from trl import SFTTrainer

# 1. Setup
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base")

# 2. Carregar dataset
loader = DatasetLoader(tokenizer)
config = DatasetConfig(name="hpcgroup/hpc-instruct", filter_language="Cuda")
dataset = loader.load_single_dataset(config)

# 3. Treinar
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    # ... outros argumentos
)
trainer.train()
```

---

## 🎁 Datasets Adicionais Opcionais

Além do dataset padrão (HPC-Instruct), o sistema suporta datasets adicionais pré-configurados que podem ser ativados via argumentos CLI.

### Datasets Disponíveis

#### 1. Evol-Instruct-Code-80k-v1
- **HuggingFace**: `nickrosh/Evol-Instruct-Code-80k-v1`
- **Descrição**: Dataset de código evolutivo com 80k exemplos
- **Colunas**: `instruction` (entrada) e `output` (resposta)
- **Uso**: Excelente para melhorar a capacidade de geração de código geral

#### 2. Magicoder-OSS-Instruct-75K
- **HuggingFace**: `ise-uiuc/Magicoder-OSS-Instruct-75K`
- **Descrição**: Dataset de instruções OSS com 75k exemplos
- **Colunas**: `problem` (entrada) e `solution` (resposta)
- **Uso**: Focado em problemas de código open-source

### Como Usar Datasets Opcionais

#### Argumentos CLI

Adicione as seguintes flags aos seus comandos de fine-tuning:

- `--use_evol_instruct`: Adiciona o dataset Evol-Instruct-Code-80k-v1
- `--use_magicoder`: Adiciona o dataset Magicoder-OSS-Instruct-75K

#### Exemplos de Uso

**1. Usar apenas o dataset padrão (HPC-Instruct)**
```bash
python finetune_deepseek_optimized.py \
    --model_name deepseek-ai/deepseek-coder-6.7b-base \
    --output_dir ./results \
    --epochs 3
```

**2. Adicionar Evol-Instruct ao treinamento**
```bash
python finetune_deepseek_optimized.py \
    --model_name deepseek-ai/deepseek-coder-6.7b-base \
    --output_dir ./results \
    --epochs 3 \
    --use_evol_instruct
```

**3. Adicionar Magicoder ao treinamento**
```bash
python finetune_deepseek_optimized.py \
    --model_name deepseek-ai/deepseek-coder-6.7b-base \
    --output_dir ./results \
    --epochs 3 \
    --use_magicoder
```

**4. Usar todos os datasets (HPC-Instruct + Evol-Instruct + Magicoder)**
```bash
python finetune_deepseek_optimized.py \
    --model_name deepseek-ai/deepseek-coder-6.7b-base \
    --output_dir ./results \
    --epochs 3 \
    --use_evol_instruct \
    --use_magicoder
```

**5. Limitar o número total de amostras**
```bash
python finetune_deepseek_optimized.py \
    --model_name deepseek-ai/deepseek-coder-6.7b-base \
    --output_dir ./results \
    --epochs 3 \
    --use_evol_instruct \
    --use_magicoder \
    --max_samples 30000
```
**Nota**: Com `--max_samples 30000` e 3 datasets, cada dataset contribuirá com ~10k amostras.

### Comportamento de Combinação

#### Distribuição de Amostras

Quando múltiplos datasets são usados:

1. **Sem limite (`--max_samples`)**: Todos os exemplos de cada dataset são usados
2. **Com limite (`--max_samples N`)**: As amostras são divididas igualmente entre os datasets
   - Exemplo: `--max_samples 30000` com 3 datasets = 10k amostras por dataset

#### Ordem de Processamento

1. Dataset principal (especificado por `--dataset_name`, padrão: `hpcgroup/hpc-instruct`)
2. Evol-Instruct (se `--use_evol_instruct` estiver ativo)
3. Magicoder (se `--use_magicoder` estiver ativo)

#### Formatação Unificada

Todos os datasets são formatados usando o mesmo template:
```
### Instruction:
{instruction}

### Response:
{response}
```

O `dataset_loader.py` automaticamente mapeia as colunas corretas de cada dataset:
- **HPC-Instruct**: `problem statement` → `solution`
- **Evol-Instruct**: `instruction` → `output`
- **Magicoder**: `problem` → `solution`

### Scripts Compatíveis

Os seguintes scripts suportam os datasets opcionais:

1. ✅ `finetune_deepseek_optimized.py`
2. ✅ `finetune_with_args.py`
3. ✅ `finetune_with_args_QLORA.py`

### Considerações de Memória

#### Impacto no VRAM

Adicionar datasets não aumenta o uso de VRAM durante o treinamento, mas:
- Aumenta o tempo de carregamento inicial
- Aumenta o número total de steps por época
- Pode requerer mais RAM do sistema para cache

#### Recomendações

| Configuração | Datasets Recomendados | Max Samples |
|--------------|----------------------|-------------|
| **Teste Rápido** | Apenas HPC-Instruct | 1000-5000 |
| **Treinamento Balanceado** | HPC + 1 adicional | 20000-50000 |
| **Treinamento Completo** | Todos os 3 | Sem limite |
| **RTX 4090 (24GB)** | Todos os 3 | Sem limite |

### Logs de Datasets Opcionais

Durante o carregamento, você verá logs como:

```
============================================================
Carregando 3 dataset(s)
============================================================

✓ Adicionando dataset opcional: Evol-Instruct-Code-80k-v1 - Dataset de código evolutivo com 80k exemplos
✓ Adicionando dataset opcional: Magicoder-OSS-Instruct-75K - Dataset de instruções OSS com 75k exemplos

[1/3] Processando: hpcgroup/hpc-instruct
Dataset 'hpcgroup/hpc-instruct' carregado
Total de exemplos: 15000
✓ Dataset 'hpcgroup/hpc-instruct' processado: 15000 exemplos

[2/3] Processando: nickrosh/Evol-Instruct-Code-80k-v1
Dataset 'nickrosh/Evol-Instruct-Code-80k-v1' carregado
Total de exemplos: 80000
✓ Dataset 'nickrosh/Evol-Instruct-Code-80k-v1' processado: 80000 exemplos

[3/3] Processando: ise-uiuc/Magicoder-OSS-Instruct-75K
Dataset 'ise-uiuc/Magicoder-OSS-Instruct-75K' carregado
Total de exemplos: 75000
✓ Dataset 'ise-uiuc/Magicoder-OSS-Instruct-75K' processado: 75000 exemplos

============================================================
Combinando 3 datasets...
✓ Dataset combinado: 170000 exemplos totais

Distribuição:
  - Dataset 1: 15000 exemplos (8.8%)
  - Dataset 2: 80000 exemplos (47.1%)
  - Dataset 3: 75000 exemplos (44.1%)
============================================================
```

### Troubleshooting - Datasets Opcionais

#### Erro: "No column names found"
- **Causa**: O dataset pode ter uma estrutura diferente
- **Solução**: Verifique as colunas do dataset no HuggingFace e ajuste `PREDEFINED_DATASETS` em `dataset_loader.py`

#### Erro: "Out of memory during loading"
- **Causa**: RAM insuficiente para carregar todos os datasets
- **Solução**: Use `--max_samples` para limitar o número de exemplos

#### Carregamento muito lento
- **Causa**: Datasets grandes sendo baixados pela primeira vez
- **Solução**: Os datasets serão cacheados após o primeiro download

### Personalização Avançada

Para adicionar novos datasets opcionais, edite `dataset_loader.py`:

```python
PREDEFINED_DATASETS = {
    'meu-dataset': {
        'name': 'usuario/nome-dataset',
        'instruction_column': 'input',
        'response_column': 'output',
        'description': 'Descrição do dataset'
    }
}
```

Depois adicione o argumento correspondente nos scripts de fine-tuning:
```python
parser.add_argument(
    "--use_meu_dataset",
    action="store_true",
    help="Adicionar meu dataset personalizado"
)
```

E modifique a função `load_dataset_from_args()` para detectar o novo argumento:
```python
if hasattr(args, 'use_meu_dataset') and args.use_meu_dataset:
    config = PREDEFINED_DATASETS['meu-dataset']
    logger.info(f"✓ Adicionando dataset opcional: {config['description']}")
    additional_datasets.append(config)
```

---

**Versão:** 1.1  
**Data:** 2025-10-07  
**Autor:** TCC - Fine-tuning DeepSeek-Coder
