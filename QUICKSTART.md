# Início rápido

[Visão geral do TCC](readme.md) · [Metodologia](docs/metodologia.md) · [Guia de reprodução](docs/reproducibilidade.md)

## Preparar o ambiente

Execute na raiz do projeto, em uma máquina com GPU NVIDIA e CUDA compatíveis:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 utils/check_compatibility.py
```

As dependências não estão fixadas em versões. Consulte os [limites de reprodução](docs/reproducibilidade.md#limites-e-registros-necessários) antes de um treinamento completo.

## Treinar com todos os datasets

O experimento descrito no TCC usa a variante **base**, não a variante `instruct`:

```bash
python3 finetune_with_args.py \
  --model_name deepseek-ai/deepseek-coder-1.3b-base \
  --output_dir ./checkpoints/gppd-full \
  --save_dir ./finetuned-models/gppd-full \
  --log_file ./logs/gppd-full.log \
  --use_evol_instruct \
  --use_magicoder
```

Para treinar apenas com HPC-INSTRUCT filtrado para CUDA, omita `--use_evol_instruct` e `--use_magicoder` e use caminhos de saída próprios para a variante HPC-only.

## Parâmetros principais

| Argumento | Função | Padrão |
| --- | --- | --- |
| `--model_name` | Modelo base no Hugging Face ou caminho local | Obrigatório |
| `--output_dir` | Checkpoints intermediários | `./results` |
| `--save_dir` | Adaptadores e tokenizer finais; tentativa de exportação mesclada | `./trained_model` |
| `--log_file` | Arquivo de log | Somente terminal |
| `--epochs` | Épocas | 3 |
| `--per_device_train_batch_size` | Batch por dispositivo | 4 |
| `--gradient_accumulation_steps` | Acumulação de gradiente | 4 |
| `--dataset_name` | Dataset principal | `hpcgroup/hpc-instruct` |
| `--filter_language` | Filtro no dataset principal | `Cuda` |

O parser também aceita `--max_length`, mas o código atual não o repassa ao treinador. Não conte com esse argumento para controlar memória sem antes ajustar a implementação. A configuração de *instruction masking* citada na monografia também não está explícita no código disponível.

## Arquivos de saída

Com o comando acima:

```text
finetuned-models/gppd-full/
├── lora_adapters/       # Adaptadores salvos pelo treinador
├── merged_model/        # Disponível se a mesclagem e exportação tiverem sucesso
└── arquivos do tokenizer
```

Os checkpoints intermediários ficam em `checkpoints/gppd-full/`. O diretório `finetuned-models/` está ignorado pelo Git; logs e checkpoints nesses outros caminhos não estão.

## Acompanhar a execução

```bash
tail -f ./logs/gppd-full.log
python3 utils/monitor_gpu.py
```

Em caso de falta de memória, reduza o batch por dispositivo e avalie a acumulação de gradiente para manter o batch efetivo desejado. Registre qualquer alteração nos parâmetros do experimento.

## Consultar os resultados existentes

Os experimentos concluídos estão documentados em [Resultados obtidos](docs/resultados.md), com tabelas calculadas dos CSVs de `metrics/`. Não é necessário treinar novamente para consultar esses resultados.

Para gerar respostas com um modelo local ou executar o protocolo do ParEval, siga o [guia de reprodução](docs/reproducibilidade.md).
