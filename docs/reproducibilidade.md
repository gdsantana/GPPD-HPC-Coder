# Reprodução dos experimentos

[Voltar ao README](../readme.md) · [Metodologia](metodologia.md)

Este guia distingue comandos do código disponível e o protocolo histórico descrito no capítulo 4 da [monografia](../TCC-GPPD-HPC-Coder.pdf). Não é uma garantia de repetição exata dos resultados: faltam versões e artefatos das execuções originais.

## Preparar o ambiente

Execute na raiz deste repositório, em uma máquina com GPU NVIDIA e ambiente CUDA compatível:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 utils/check_compatibility.py
```

As dependências não têm versões fixadas. O diagnóstico é auxiliar e não comprova compatibilidade de todas as APIs. Antes de um treinamento longo, valide carregamento de modelo, dados e um pequeno treino no ambiente escolhido. A instalação de `flash-attn` pode exigir ferramentas de compilação e configuração específica do CUDA.

## Treinar as variantes

### Full

```bash
python3 finetune_with_args.py \
  --model_name deepseek-ai/deepseek-coder-1.3b-base \
  --output_dir ./checkpoints/gppd-full \
  --save_dir ./finetuned-models/gppd-full \
  --log_file ./logs/gppd-full.log \
  --use_evol_instruct \
  --use_magicoder
```

### HPC-only

```bash
python3 finetune_with_args.py \
  --model_name deepseek-ai/deepseek-coder-1.3b-base \
  --output_dir ./checkpoints/gppd-hpc-only \
  --save_dir ./finetuned-models/gppd-hpc-only \
  --log_file ./logs/gppd-hpc-only.log
```

`--output_dir` recebe os checkpoints intermediários; `--save_dir`, o resultado final. O script salva `lora_adapters/` e o tokenizer no diretório final. Ele também tenta salvar `merged_model/`, dependendo do sucesso da mesclagem. Apenas `finetuned-models/`, `venv`, `__pycache__` e `.vscode` estão no `.gitignore` atual; revise os demais artefatos antes de um commit.

## Gerar respostas localmente

Para um modelo mesclado exportado com sucesso:

```bash
python3 utils/generate_prompts.py \
  --model_dir ./finetuned-models/gppd-full/merged_model \
  --prompts_dir ./prompts \
  --output_dir ./outputs/gppd-full \
  --k 3 \
  --seed 42
```

Esse utilitário lê os arquivos de prompts recursivamente e grava respostas `cuda.gen-001.txt`, por exemplo, mantendo a hierarquia das categorias. É um exemplo de inferência; não executa testes nem substitui o fluxo histórico de avaliação. O TCC utilizou a geração do ParEval, com 50 respostas por prompt, não este exemplo com três respostas.

## Executar o benchmark histórico

Os comandos desta seção são transcritos e adaptados das Listagens 4.2 a 4.6. Devem ser executados na raiz do [ParEval usado no trabalho](https://github.com/gdsantana/ParEval), após instalar e compilar suas dependências. Os scripts e arquivos de configuração abaixo não pertencem a este repositório.

Primeiro, ajuste o compilador em `drivers/build-configs.json` ao ambiente e prepare `prompts/cuda-only-prompts.json` filtrando os prompts CUDA do benchmark. Esse JSON não está incluído no acervo local. Os caminhos entre `<...>` são marcadores a substituir; o formato dos arquivos intermediários depende da versão do ParEval.

```bash
python3 generate/generate.py \
  --prompts prompts/cuda-only-prompts.json \
  --model <modelo-ou-diretorio> \
  --output <geracoes.json> \
  --cache <cache.jsonl> \
  --do_sample \
  --prompted
```

A monografia relata 50 respostas por prompt por padrão nessa execução. Confirme esse padrão na revisão utilizada e registre os parâmetros de amostragem. Para modelos de 6,7B, o trabalho acrescenta `--batch_size 4`.

```bash
python3 drivers/run-all.py <geracoes.json> \
  -o <resultado-execucao> \
  --include-models cuda \
  --launch-configs drivers/launch-configs.json \
  --build-configs drivers/build-configs.json \
  --problem-sizes drivers/problem-sizes.json \
  --yes-to-all

python3 analysis/create-dataframe.py <resultado-execucao> \
  --output <dataframe>

python3 analysis/metrics.py <dataframe> \
  --output <metricas> \
  --problem-sizes drivers/problem-sizes.json \
  --model-name <identificador-do-modelo>
```

## Consultar agregados e gerar gráficos

É possível recalcular as médias documentadas usando apenas a biblioteca padrão do Python:

```bash
python3 - <<'PY'
import csv
from collections import defaultdict

groups = defaultdict(list)
with open('metrics/all-metrics.csv', encoding='utf-8') as source:
    for row in csv.DictReader(source):
        if row['model'].strip():
            groups[row['model']].append(row)
for model, rows in groups.items():
    means = {
        metric: round(100 * sum(float(row[metric]) for row in rows) / len(rows), 2)
        for metric in ('build@1', 'pass@1', 'pass@5', 'pass@10', 'pass@20')
    }
    print(model, means)
PY
```

Para gerar figuras com pandas e Matplotlib, ajuste `INPUT_FILE` e `OUTPUT_FILE` em [generate_chart.py](../generate_chart.py) ou [generate_line_chart.py](../generate_line_chart.py), depois execute o script correspondente. O primeiro produz barras por categoria; o segundo calcula médias por modelo para os valores de k. Configure um novo caminho de saída para preservar figuras históricas. Esses scripts não recriam automaticamente todas as coleções `Graficos*`.

## Limites e registros necessários

Antes de afirmar reprodução integral, é necessário recuperar ou registrar:

1. **Versões:** commit deste projeto e do ParEval, Python, pacotes, CUDA, driver e compilador.
2. **Treinamento:** revisão do modelo base e datasets, quantidade de exemplos por fonte, sementes, comando completo, logs e pesos/adaptadores finais.
3. **Avaliação:** prompts JSON, respostas individuais, cache, configurações de compilação e execução, tamanhos de problema, dataframe e saídas brutas.
4. **Métricas:** fórmula e agregação de speedup/eficiência, tratamento de falhas e filtros efetivamente aplicados.

Também há diferenças entre a descrição e a implementação atual:

- O TCC menciona *instruction masking*, mas não há configuração explícita de máscara de perda no treinador disponível.
- `--max_length` é aceito pelo parser, mas não é repassado ao `SFTTrainer`; o efeito anunciado de ajustar esse argumento não está implementado explicitamente.
- O texto de treinamento muda entre as variantes: HPC-only usa `Instruction: ...`, e a inclusão de datasets adicionais usa `### Instruction: ...`. Logo, o experimento atual não altera apenas a composição dos dados.
- O carregador pode continuar com os datasets que foram carregados com sucesso quando outro falha. Confira as fontes e contagens reais nos logs para identificar a variante treinada.

Esses pontos foram documentados sem alterar o código de treinamento ou os resultados históricos.
