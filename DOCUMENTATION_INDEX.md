# 📚 Índice de Documentação

Guia completo de toda a documentação disponível no projeto.

## 🎯 Por Onde Começar?

```
┌─────────────────────────────────────────────────────────┐
│  Novo no projeto? Comece aqui:                          │
│                                                         │
│  1. readme.md          - Visão geral e quick start      │
│  2. QUICKSTART.md      - Comandos essenciais            │
│  3. check_compatibility.py - Verificar seu sistema      │
│  4. Treinar seu primeiro modelo!                        │
└─────────────────────────────────────────────────────────┘
```

## 📖 Documentação Principal

### 1. **[readme.md](readme.md)** 
**Visão Geral do Projeto**
- ⚡ Quick start em 4 passos
- 🆕 Novidades (módulo dataset_loader)
- 🛠️ Lista de scripts principais
- 📖 Exemplos básicos de uso
- 🎯 Configurações recomendadas

**Quando usar:** Primeira vez no projeto ou referência rápida

---

### 2. **[README_DEEPSEEK_FINETUNE.md](README_DEEPSEEK_FINETUNE.md)**
**Documentação Completa e Técnica**
- 🎯 Otimizações implementadas (QLoRA, Flash Attention, etc)
- 📋 Requisitos de hardware e software
- 🚀 Exemplos de uso detalhados
- ⚙️ Tabela completa de parâmetros
- 📊 Monitoramento de memória
- 🎛️ Ajuste fino de hiperparâmetros
- 💾 Estimativas de uso de memória
- 🔧 Guia de inferência
- 🐛 Troubleshooting completo
- 📈 Benchmarks de performance

**Quando usar:** Configuração avançada, otimização, troubleshooting

---

### 3. **[QUICKSTART.md](QUICKSTART.md)**
**Guia de Início Rápido**
- ⚡ Setup em minutos
- 🎯 Comandos prontos para copiar
- 🔧 Configurações alternativas
- 💡 Troubleshooting básico

**Quando usar:** Quer começar rapidamente sem ler muita documentação

---

### 4. **[DATASET_LOADER_GUIDE.md](DATASET_LOADER_GUIDE.md)** 🆕
**Guia Completo do Módulo Dataset Loader**
- 🎯 Visão geral do módulo
- 📦 Componentes (DatasetConfig, DatasetLoader)
- 🚀 Exemplos práticos (10+ exemplos)
- 🔧 Argumentos CLI suportados
- 📊 Fallbacks automáticos
- 🎨 Templates de formatação
- 🔍 Tratamento de erros
- 🔄 Guia de migração
- 🎓 Casos de uso comuns
- 🐛 Troubleshooting

**Quando usar:** Trabalhar com datasets, múltiplos datasets, formatos customizados

---

### 5. **[REFACTORING_NOTES.md](REFACTORING_NOTES.md)** 🆕
**Notas de Refatoração**
- 📋 Resumo das mudanças
- 🎯 Objetivos alcançados
- 🔧 Scripts refatorados
- 📊 Comparações antes/depois
- 🆕 Novas capacidades
- 🔍 Detalhes técnicos
- 🎯 Benefícios
- 🚀 Próximos passos
- 📚 Exemplos de migração
- ✅ Checklist

**Quando usar:** Entender as mudanças recentes, migrar código antigo

---

## 🛠️ Documentação de Scripts

### Scripts de Treinamento

#### **finetune_deepseek_optimized.py**
Script principal com todas as otimizações avançadas.

**Documentação relevante:**
- [README_DEEPSEEK_FINETUNE.md](README_DEEPSEEK_FINETUNE.md) - Seção "Uso Básico"
- [QUICKSTART.md](QUICKSTART.md) - Comando recomendado
- [DATASET_LOADER_GUIDE.md](DATASET_LOADER_GUIDE.md) - Integração com datasets

**Exemplo:**
```bash
python finetune_deepseek_optimized.py \
  --model_name deepseek-ai/deepseek-coder-6.7b-base \
  --epochs 3 \
  --use_flash_attention
```

---

#### **finetune_with_args.py**
Script configurável via argumentos CLI.

**Documentação relevante:**
- [DATASET_LOADER_GUIDE.md](DATASET_LOADER_GUIDE.md) - Exemplo 5
- [REFACTORING_NOTES.md](REFACTORING_NOTES.md) - Seção "Scripts Refatorados"

**Exemplo:**
```bash
python finetune_with_args.py \
  --model_name deepseek-ai/deepseek-coder-1.3b-base \
  --dataset_name hpcgroup/hpc-instruct \
  --filter_language Cuda
```

---

### Scripts Utilitários

#### **check_compatibility.py**
Verificador de compatibilidade do sistema.

**Documentação relevante:**
- [README_DEEPSEEK_FINETUNE.md](README_DEEPSEEK_FINETUNE.md) - Seção "Requisitos"
- [QUICKSTART.md](QUICKSTART.md) - Passo 2

**Uso:**
```bash
python check_compatibility.py
```

---

#### **monitor_gpu.py**
Monitor de memória GPU em tempo real.

**Documentação relevante:**
- [README_DEEPSEEK_FINETUNE.md](README_DEEPSEEK_FINETUNE.md) - Seção "Monitoramento de Memória"

**Uso:**
```bash
python monitor_gpu.py
python monitor_gpu.py --no-cpu
python monitor_gpu.py --once
```

---

#### **inference_example.py**
Script de inferência com modelos treinados.

**Documentação relevante:**
- [README_DEEPSEEK_FINETUNE.md](README_DEEPSEEK_FINETUNE.md) - Seção "Inferência"
- [QUICKSTART.md](QUICKSTART.md) - Seção "Inferência"

**Uso:**
```bash
python inference_example.py --adapter_path ./trained_model --interactive
```

---

#### **generate_prompts.py**
Gerador de prompts em lote.

**Documentação relevante:**
- [readme.md](readme.md) - Seção "Gerar Prompts"

**Uso:**
```bash
python generate_prompts.py \
  --model_dir ./trained_model \
  --prompts_dir ./prompts \
  --output_dir ./outputs
```

---

### Módulos

#### **dataset_loader.py** 🆕
Módulo modular para carregamento de datasets.

**Documentação relevante:**
- [DATASET_LOADER_GUIDE.md](DATASET_LOADER_GUIDE.md) - Documentação completa
- [REFACTORING_NOTES.md](REFACTORING_NOTES.md) - Contexto da refatoração

**Uso:**
```python
from dataset_loader import DatasetConfig, DatasetLoader

loader = DatasetLoader(tokenizer)
config = DatasetConfig(name="hpcgroup/hpc-instruct")
dataset = loader.load_single_dataset(config)
```

---

## 🎓 Guias por Cenário

### Cenário 1: Primeiro Treinamento
```
1. readme.md (visão geral)
2. QUICKSTART.md (comandos)
3. check_compatibility.py (verificar sistema)
4. Executar treinamento básico
```

### Cenário 2: Otimizar Performance
```
1. README_DEEPSEEK_FINETUNE.md (seção "Otimizações")
2. README_DEEPSEEK_FINETUNE.md (seção "Ajuste Fino")
3. README_DEEPSEEK_FINETUNE.md (seção "Uso de Memória")
4. Ajustar hiperparâmetros
```

### Cenário 3: Trabalhar com Datasets
```
1. DATASET_LOADER_GUIDE.md (visão geral)
2. DATASET_LOADER_GUIDE.md (exemplos práticos)
3. DATASET_LOADER_GUIDE.md (casos de uso)
4. Implementar seu dataset
```

### Cenário 4: Múltiplos Datasets
```
1. DATASET_LOADER_GUIDE.md (exemplo 4)
2. DATASET_LOADER_GUIDE.md (seção "Como Funciona")
3. README_DEEPSEEK_FINETUNE.md (seção "Múltiplos Datasets")
4. Configurar e treinar
```

### Cenário 5: Troubleshooting
```
1. README_DEEPSEEK_FINETUNE.md (seção "Troubleshooting")
2. DATASET_LOADER_GUIDE.md (seção "Troubleshooting")
3. QUICKSTART.md (seção "Troubleshooting")
4. Resolver problema
```

### Cenário 6: Migrar Código Antigo
```
1. REFACTORING_NOTES.md (visão geral)
2. REFACTORING_NOTES.md (exemplos de migração)
3. DATASET_LOADER_GUIDE.md (guia de migração)
4. Refatorar código
```

---

## 📊 Mapa Mental da Documentação

```
TCC Documentation
│
├── 🚀 Getting Started
│   ├── readme.md
│   ├── QUICKSTART.md
│   └── check_compatibility.py
│
├── 📖 Technical Documentation
│   ├── README_DEEPSEEK_FINETUNE.md
│   │   ├── Optimizations
│   │   ├── Parameters
│   │   ├── Memory Management
│   │   ├── Inference
│   │   └── Troubleshooting
│   │
│   └── DATASET_LOADER_GUIDE.md 🆕
│       ├── API Reference
│       ├── Examples
│       ├── Use Cases
│       └── Migration Guide
│
├── 🔄 Refactoring
│   └── REFACTORING_NOTES.md 🆕
│       ├── Changes Summary
│       ├── Before/After
│       └── Benefits
│
└── 🛠️ Scripts Documentation
    ├── Training Scripts
    ├── Utility Scripts
    └── Modules
```

---

## 🔍 Busca Rápida

### Por Tópico

**Datasets:**
- [DATASET_LOADER_GUIDE.md](DATASET_LOADER_GUIDE.md) - Guia completo
- [REFACTORING_NOTES.md](REFACTORING_NOTES.md) - Contexto

**Otimização:**
- [README_DEEPSEEK_FINETUNE.md](README_DEEPSEEK_FINETUNE.md) - Seções 1-7
- [QUICKSTART.md](QUICKSTART.md) - Configurações

**Troubleshooting:**
- [README_DEEPSEEK_FINETUNE.md](README_DEEPSEEK_FINETUNE.md) - Seção "Troubleshooting"
- [DATASET_LOADER_GUIDE.md](DATASET_LOADER_GUIDE.md) - Seção "Troubleshooting"

**Exemplos:**
- [DATASET_LOADER_GUIDE.md](DATASET_LOADER_GUIDE.md) - 10+ exemplos
- [readme.md](readme.md) - Exemplos básicos
- [QUICKSTART.md](QUICKSTART.md) - Comandos prontos

**API Reference:**
- [DATASET_LOADER_GUIDE.md](DATASET_LOADER_GUIDE.md) - Seções 1-2

---

## 📝 Checklist de Leitura

### Para Iniciantes
- [ ] Ler readme.md
- [ ] Ler QUICKSTART.md
- [ ] Executar check_compatibility.py
- [ ] Fazer primeiro treinamento
- [ ] Ler README_DEEPSEEK_FINETUNE.md (visão geral)

### Para Desenvolvedores
- [ ] Ler DATASET_LOADER_GUIDE.md
- [ ] Ler REFACTORING_NOTES.md
- [ ] Entender arquitetura do módulo
- [ ] Implementar dataset customizado
- [ ] Contribuir com melhorias

### Para Otimização
- [ ] Ler seção de otimizações
- [ ] Entender uso de memória
- [ ] Testar diferentes configurações
- [ ] Monitorar performance
- [ ] Ajustar hiperparâmetros

---

## 🔗 Links Externos

- [DeepSeek-Coder GitHub](https://github.com/deepseek-ai/DeepSeek-Coder)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Flash Attention 2](https://arxiv.org/abs/2307.08691)

---

## 📞 Suporte

**Dúvidas sobre:**
- **Instalação/Setup:** readme.md, QUICKSTART.md
- **Configuração:** README_DEEPSEEK_FINETUNE.md
- **Datasets:** DATASET_LOADER_GUIDE.md
- **Erros:** Seções de Troubleshooting
- **Código:** REFACTORING_NOTES.md

---

**Última Atualização:** 2025-10-07  
**Versão da Documentação:** 2.0
