# Metodologia

[Voltar ao README](../readme.md)

Esta síntese reúne o capítulo 4 da [monografia](../TG2_rascunho.pdf) e a implementação disponível. Informações históricas do texto são identificadas separadamente de configurações verificáveis no código.

## Objetivo e desenho experimental

Investigar o efeito da especialização de um Small Language Model na geração de código CUDA. A comparação contempla dois refinamentos do DeepSeek-Coder-1.3B, seu modelo base e os HPC-Coder-V2 de 1,3B e 6,7B parâmetros. A avaliação se restringe a CUDA, sem estender as conclusões a MPI ou OpenMP.

## Ambiente

Segundo o TCC, treinamento e benchmark foram executados no GPPD/UFRGS, com alocação via Slurm na partição `tupi`, acesso por SSH e ambiente virtual Python. O refinamento utilizou uma NVIDIA GeForce RTX 4090 com 24 GB de memória.

As versões de Python, CUDA, driver e bibliotecas usadas originalmente não estão registradas no acervo. A lista atual em `requirements.txt` não fixa versões.

## Dados e variantes

| Variante | Dados de treinamento | Identificador nos CSVs, sem o sufixo `[v2]` |
| --- | --- | --- |
| GPPD Full | HPC-INSTRUCT filtrado para CUDA + Evol-Instruct-Code + Magicoder-OSS-Instruct | `gppd-hpc-cuda-coder-1.3b-base-full-datasets` |
| GPPD HPC-only | Somente HPC-INSTRUCT filtrado para CUDA | `gppd-hpc-cuda-coder-1.3b-base-hpc-only` |

O módulo [dataset_loader.py](../dataset_loader.py) carrega os datasets `hpcgroup/hpc-instruct`, `nickrosh/Evol-Instruct-Code-80k-v1` e `ise-uiuc/Magicoder-OSS-Instruct-75K`. O filtro padrão é `language == "Cuda"` e se aplica ao dataset principal. Os dois conjuntos adicionais mantêm exemplos de programação geral.

Os exemplos são convertidos em texto de instrução/resposta e recebem o token EOS. O módulo concatena os conjuntos carregados. O TCC informa 167.756 instruções e cerca de 13 horas de treinamento para Full; não há logs originais para reconferir essa contagem ou duração.

## Refinamento

O modelo base da Listagem 4.1 é `deepseek-ai/deepseek-coder-1.3b-base`. O script [finetune_with_args.py](../finetune_with_args.py) utiliza Transformers, PEFT, TRL/SFTTrainer e bitsandbytes.

| Configuração | Valor no código atual |
| --- | --- |
| Carregamento do modelo | Quantização de 8 bits, `torch.float16` |
| LoRA | rank 16, alpha 32, dropout 0,05, sem bias |
| Épocas | 3 por padrão |
| Batch por dispositivo | 4 por padrão |
| Acumulação de gradiente | 4 passos por padrão |
| Learning rate | `2e-4` |
| Otimizador | `paged_adamw_8bit` |
| Scheduler e warmup | Cosseno, razão de warmup 0,03 |
| Checkpoints | A cada 100 passos, mantendo até 3 |

Esses valores descrevem a implementação disponível, sem comprovar todos os parâmetros efetivamente usados em cada execução histórica. O texto menciona *instruction masking*, mas o script não configura explicitamente uma máscara de perda para as instruções. Além disso, `--max_length` é declarado, porém não é repassado ao treinador. Consulte as [limitações de reprodução](reproducibilidade.md#limites-e-registros-necessários).

## Avaliação com ParEval

Conforme a seção 4.3, o fluxo foi:

1. Ajustar o compilador CUDA em `drivers/build-configs.json`, usando `/usr/local/cuda/bin/nvcc` no ambiente experimental.
2. Filtrar o arquivo de prompts do ParEval para CUDA.
3. Gerar 50 respostas por prompt, com amostragem e `--prompted`. Para modelos de 6,7B, usar `--batch_size 4`.
4. Compilar e executar as respostas com `drivers/run-all.py`.
5. Consolidar as saídas com `analysis/create-dataframe.py` e calcular métricas com `analysis/metrics.py`.

Esses scripts pertencem ao repositório separado do ParEval. O diretório local [prompts/](../prompts/) contém 60 arquivos CUDA, distribuídos em 12 categorias de cinco problemas cada: `dense_la`, `fft`, `geometry`, `graph`, `histogram`, `reduce`, `scan`, `search`, `sort`, `sparse_la`, `stencil` e `transform`.

## Métricas e agregação

- **build@k:** estimativa de obter ao menos uma resposta compilável entre k amostras.
- **pass@k:** estimativa de obter ao menos uma resposta funcionalmente correta entre k amostras.
- **speedup e efficiency:** indicadores de execução exportados pelo benchmark; interpretar com as referências, os tamanhos de problema e as regras de agregação da execução original.

Os CSVs registram k = 1, 5, 10 e 20. As tabelas desta documentação usam a média aritmética das 12 categorias, seguindo a agregação por modelo dos scripts de gráficos. Como cada categoria tem cinco prompts locais, a distribuição local é equilibrada. Não foram calculados intervalos de confiança nem realizados novos benchmarks.
