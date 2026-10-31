# Organização do acervo

[Voltar ao README](../readme.md)

## Organização adotada

O README apresenta o trabalho e suas conclusões. A pasta `docs/` aprofunda [metodologia](metodologia.md), [resultados](resultados.md) e [reprodução](reproducibilidade.md). Os arquivos de código, CSVs e figuras conservam seus caminhos para manter referências e scripts existentes.

| Local | Função |
| --- | --- |
| `TCC-GPPD-HPC-Coder.pdf` | Texto da monografia associado ao projeto. |
| `finetune_with_args.py`, `dataset_loader.py` | Implementação de treinamento e preparação dos dados. |
| `utils/` | Diagnósticos, monitoramento, inferência e geração local de respostas. |
| `prompts/` | 60 prompts CUDA em 12 categorias; não contém o JSON de entrada do ParEval. |
| `metrics/` | 13 CSVs de resultados consolidados e recortes. |
| `graphs/` | Figuras de compilação e correção relacionadas à discussão principal. |
| `Graficos/` | Comparações por categoria, incluindo speedup e eficiência. |
| `Graficos GPPD-HPC/` | Coleção adicional de curvas build, pass, speedup e eficiência. |
| `Graficos HPC-Coder-V2xGPPD-Coder/` | Coleção adicional de comparações build e pass. |
| `TCC.png` | Imagem avulsa disponível, sem vínculo experimental documentado. |

## Catálogo das métricas

Todos os caminhos abaixo são relativos a `metrics/`:

| Arquivo ou padrão | Conteúdo e uso |
| --- | --- |
| `all-metrics.csv` | Cinco modelos, 12 categorias, seis métricas para k = 1, 5, 10 e 20. Fonte da síntese documental. |
| `build@1-gppd.csv`, `pass@1-gppd.csv` | Comparação do modelo base e dos dois GPPD por categoria. O arquivo de pass também inclui outros valores de k. |
| `build@1-all-models.csv`, `pass@1-all-models.csv` | Comparação dos dois GPPD e dos dois HPC-Coder-V2 por categoria; não inclui o modelo base. |
| `build@k-gppd-average.csv`, `pass@k-gppd-average.csv` | Dados por categoria para calcular a média do modelo base e dos GPPD. Apesar de `average` no nome, ainda precisam de agregação. |
| `build@k-all-models.csv`, `pass@k-all-models.csv` | Dados para as curvas dos quatro modelos refinados. O CSV de build não inclui nomes das categorias. |
| `speedup@1.csv`, `efficiency@1.csv` | Indicadores de execução por categoria para cinco modelos. |
| `speedup@1-without-stencil.csv`, `efficiency@1-without-stencil.csv` | Recortes para quatro modelos, mas ainda com linhas `stencil`; conferir o conteúdo antes de usar. |

Os identificadores com `[v2]` aparecem no consolidado e nos arquivos de desempenho; os recortes de build/pass omitem esse sufixo. As métricas de build/pass foram conferidas entre esses arquivos, sem renomear os dados originais.

## Mapa das figuras do capítulo 5

Os arquivos em `graphs/` correspondem aos temas e comparações das figuras da monografia:

| Figura | Comparação | Arquivo em `graphs/` |
| --- | --- | --- |
| 5.1 | build@k: base e GPPD | [build@k-gppd-average.png](../graphs/build@k-gppd-average.png) |
| 5.2 | build@1 por categoria: base e GPPD | [build@1-gppd.png](../graphs/build@1-gppd.png) |
| 5.3 | pass@k: base e GPPD | [pass@k-gppd-average.png](../graphs/pass@k-gppd-average.png) |
| 5.4 | pass@1 por categoria: base e GPPD | [pass@1-gppd.png](../graphs/pass@1-gppd.png) |
| 5.5 | build@k: GPPD e HPC-Coder-V2 | [build@k-all-models.png](../graphs/build@k-all-models.png) |
| 5.6 | build@1 por categoria: GPPD e HPC-Coder-V2 | [build@1-all-models.png](../graphs/build@1-all-models.png) |
| 5.7 | pass@k: GPPD e HPC-Coder-V2 | [pass@k-all-models.png](../graphs/pass@k-all-models.png) |
| 5.8 | pass@1 por categoria: GPPD e HPC-Coder-V2 | [pass@1-all-models.png](../graphs/pass@1-all-models.png) |

As outras coleções de figuras foram preservadas. Não se deve presumir que cada imagem dessas coleções foi produzida pelos dois scripts atuais ou com os mesmos filtros.

## Próximos passos recomendados

Estas são propostas para completar o acervo, não movimentações já realizadas:

1. Confirmar a versão definitiva da monografia e reunir os slides da apresentação, se disponíveis, em uma área de documentos acadêmicos.
2. Recuperar pesos/adaptadores e publicar sua localização e revisão; manter os arquivos grandes em armazenamento apropriado, com links no README.
3. Reunir um manifesto por experimento, com comando, ambiente, datasets, sementes e referências para saídas brutas do ParEval.
4. Após identificar a origem das figuras, consolidar as coleções em `results/figures/`, com métricas em `results/metrics/` e scripts atualizados na mesma mudança. Manter um mapa dos nomes antigos para os novos.
5. Arquivar os dados brutos por execução, separados dos agregados publicados, e definir uma licença para o código de autoria própria.

Para novas execuções, usar identificadores que distingam variante, data e revisão. Registrar em cada execução a entrada, os parâmetros e o caminho do resultado evita misturar versões do experimento.
