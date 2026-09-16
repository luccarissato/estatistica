# Relatório Executivo

## Análise Exploratória dos Dados do Austin Animal Center

## Parte 1 - Introdução

Este projeto realiza uma análise exploratória dos dados do Austin Animal Center, um abrigo municipal localizado em Austin, Texas, nos Estados Unidos. A base utilizada é o arquivo `aac_shelter_outcomes.csv`, que registra informações sobre os animais no momento de sua saída do abrigo, incluindo espécie, idade, raça, cor, sexo e tipo de desfecho.

O objetivo da análise é compreender como algumas características dos animais se relacionam com os resultados registrados pelo abrigo. Entre os principais desfechos presentes na base estão adoção, transferência, retorno ao tutor, eutanásia e outros casos menos frequentes. A partir dessas informações, o projeto busca transformar os dados em respostas estatísticas claras para um problema real.

As perguntas investigativas que orientam o trabalho são:

1. Qual espécie apresenta maior proporção de adoções entre seus respectivos desfechos?
2. A idade dos animais adotados difere da idade dos animais transferidos?
3. Quais grupos de desfecho apresentam maior variabilidade de idade e maior presença de valores extremos?

Com base nessas perguntas, o relatório pretende validar ou rejeitar as hipóteses levantadas inicialmente: que a proporção de adoções pode variar entre espécies, que animais adotados e transferidos podem apresentar distribuições de idade diferentes, e que alguns tipos de desfecho podem concentrar maior dispersão e mais outliers em relação à idade.

## Parte 2 - Entendimento e pré-processamento da base

Nesta etapa, carregamos a base `aac_shelter_outcomes.csv`, verificamos sua estrutura inicial e preparamos os dados para as análises estatísticas posteriores. A base original possui 78.256 registros e 12 colunas, contendo informações sobre idade, tipo de animal, raça, cor, datas, nome, tipo de desfecho e sexo do animal.

O carregamento é feito a partir do arquivo CSV armazenado na pasta `data`:

```python
from pathlib import Path
import re

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "aac_shelter_outcomes.csv"
OUTPUT_GRAPHS_DIR = BASE_DIR / "outputs" / "graficos"
OUTPUT_TABLES_DIR = BASE_DIR / "outputs" / "tabelas"


def carregar_base():
    # Carrega a base principal do projeto: registros de saida dos animais
    # do Austin Animal Center.
    return pd.read_csv(DATA_PATH)
```

Depois do carregamento, o código verifica a quantidade de linhas e colunas, os nomes das colunas, os tipos de dados, as primeiras linhas e as frequências iniciais de `animal_type` e `outcome_type`.

```python
def entender_base(df):
    print("=" * 80)
    print("1. PREPARACAO E ENTENDIMENTO DA BASE")
    print("=" * 80)

    print("\nQuantidade de linhas e colunas:")
    print(df.shape)

    print("\nColunas disponiveis:")
    print(df.columns.tolist())

    print("\nTipos de dados:")
    print(df.dtypes)

    print("\nPrimeiras linhas da base:")
    print(df.head())

    print("\nFrequencia por tipo de animal:")
    print(df["animal_type"].value_counts())

    print("\nFrequencia por tipo de desfecho:")
    print(df["outcome_type"].value_counts(dropna=False))
```

Na análise inicial, os tipos de animal mais frequentes foram cães, gatos e outros animais. Em relação aos desfechos, os mais comuns foram adoção, transferência e retorno ao tutor. Esses resultados ajudam a entender o perfil geral da base e indicam caminhos possíveis para as perguntas investigativas.

Antes do tratamento, foram encontrados valores ausentes principalmente em `name`, `outcome_subtype`, `outcome_type`, `age_upon_outcome` e `sex_upon_outcome`. Também foram identificadas 10 linhas completamente duplicadas.

A idade aparece originalmente como texto, em formatos como `2 weeks`, `1 year` e `5 months`. Por isso, foi criada uma função para converter a idade para dias:

```python
def converter_idade_para_dias(idade):
    # Converte valores como "2 weeks", "1 year" e "5 months" para dias.
    # A conversao para uma unidade numerica comum e necessaria para calcular
    # media, mediana, quartis, variancia, desvio padrao e correlacao.
    if pd.isna(idade):
        return pd.NA

    resultado = re.search(r"(\d+)\s+(\w+)", str(idade).strip().lower())

    if resultado is None:
        return pd.NA

    valor = int(resultado.group(1))
    unidade = resultado.group(2)

    if unidade.startswith("day"):
        return valor
    if unidade.startswith("week"):
        return valor * 7
    if unidade.startswith("month"):
        return valor * 30
    if unidade.startswith("year"):
        return valor * 365

    return pd.NA
```

O pré-processamento remove duplicatas completas, converte colunas de data, cria as variáveis `age_days`, `age_years` e `outcome_year`, e remove registros sem idade numérica ou tipo de desfecho.

```python
def preprocessar_dados(df):
    print("\n" + "=" * 80)
    print("2. PRE-PROCESSAMENTO DOS DADOS")
    print("=" * 80)

    df_processado = df.copy()

    print("\nValores ausentes antes do tratamento:")
    print(df_processado.isna().sum())

    duplicados = df_processado.duplicated().sum()
    print("\nLinhas completamente duplicadas encontradas:")
    print(duplicados)

    # Remove apenas linhas completamente duplicadas, pois animal_id repetido
    # pode representar um novo registro de saida do mesmo animal.
    df_processado = df_processado.drop_duplicates().copy()

    # Converte colunas de data para permitir analises temporais posteriores.
    df_processado["date_of_birth"] = pd.to_datetime(
        df_processado["date_of_birth"], errors="coerce"
    )
    df_processado["datetime"] = pd.to_datetime(
        df_processado["datetime"], errors="coerce"
    )
    df_processado["monthyear"] = pd.to_datetime(
        df_processado["monthyear"], errors="coerce"
    )

    # Cria variaveis numericas de idade para viabilizar as analises estatisticas.
    df_processado["age_days"] = df_processado["age_upon_outcome"].apply(
        converter_idade_para_dias
    )
    df_processado["age_days"] = pd.to_numeric(
        df_processado["age_days"], errors="coerce"
    )
    df_processado["age_years"] = df_processado["age_days"] / 365

    # O ano do desfecho sera util em etapas futuras, inclusive para comparar
    # variaveis numericas em escalas diferentes na padronizacao.
    df_processado["outcome_year"] = df_processado["datetime"].dt.year

    # Remove registros sem informacoes essenciais para as perguntas do projeto:
    # idade numerica e tipo de desfecho.
    df_processado = df_processado.dropna(subset=["age_days", "outcome_type"]).copy()

    print("\nValores ausentes depois do tratamento essencial:")
    print(df_processado.isna().sum())

    print("\nTipos de dados depois das conversoes:")
    print(df_processado.dtypes)

    print("\nQuantidade de linhas e colunas depois do pre-processamento:")
    print(df_processado.shape)

    return df_processado
```

Após esse processo, a base passou para 78.228 registros e 15 colunas. Essa etapa deixa os dados prontos para calcular as métricas estatísticas exigidas no projeto.

## Parte 3 - Estatísticas descritivas e visualizações gerais

Após o pré-processamento, foi realizada uma análise descritiva geral da idade dos animais. Essa etapa tem como objetivo resumir o comportamento da variável `age_years`, que representa a idade dos animais em anos no momento do desfecho.

O código calcula medidas de centralização, posição e dispersão, atendendo aos requisitos estatísticos iniciais do projeto:

```python
def analisar_estatisticas_gerais(df):
    print("\n" + "=" * 80)
    print("3. ESTATISTICAS DESCRITIVAS E VISUALIZACOES GERAIS")
    print("=" * 80)

    OUTPUT_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    idade = df["age_years"]

    estatisticas = pd.DataFrame(
        {
            "metrica": [
                "media",
                "mediana",
                "moda",
                "q1",
                "q2",
                "q3",
                "percentil_10",
                "percentil_90",
                "variancia",
                "desvio_padrao",
                "amplitude",
            ],
            "valor": [
                idade.mean(),
                idade.median(),
                idade.mode().iloc[0],
                idade.quantile(0.25),
                idade.quantile(0.50),
                idade.quantile(0.75),
                idade.quantile(0.10),
                idade.quantile(0.90),
                idade.var(),
                idade.std(),
                idade.max() - idade.min(),
            ],
        }
    )

    print("\nEstatisticas gerais da idade em anos:")
    print(estatisticas)

    estatisticas.to_csv(
        OUTPUT_TABLES_DIR / "estatisticas_gerais_idade.csv", index=False
    )
```

Os resultados obtidos foram:

| Métrica | Valor |
|---|---:|
| Média | 2,13 anos |
| Mediana | 1,00 ano |
| Moda | 1,00 ano |
| Q1 | 0,25 ano |
| Q2 | 1,00 ano |
| Q3 | 3,00 anos |
| Percentil 10 | 0,08 ano |
| Percentil 90 | 6,00 anos |
| Variância | 8,39 |
| Desvio padrão | 2,90 anos |
| Amplitude | 25,00 anos |

A média de idade dos animais é de aproximadamente 2,13 anos, enquanto a mediana e a moda são iguais a 1 ano. Isso indica que muitos animais deixam o abrigo ainda jovens. Como a média é maior do que a mediana, a distribuição apresenta influência de animais mais velhos, que puxam a média para cima.

Os quartis mostram que 25% dos animais tinham até aproximadamente 0,25 ano, metade tinha até 1 ano e 75% tinham até 3 anos no momento do desfecho. O percentil 90 indica que 90% dos animais tinham até 6 anos. A amplitude de 25 anos e o desvio padrão de 2,90 anos mostram que existe variação relevante na idade dos animais, embora a maior parte dos registros esteja concentrada em idades mais baixas.

Além da tabela estatística, foram geradas visualizações gerais para apoiar a interpretação da base:

```python
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="animal_type", order=df["animal_type"].value_counts().index)
    plt.title("Distribuicao por tipo de animal")
    plt.xlabel("Tipo de animal")
    plt.ylabel("Quantidade de registros")
    salvar_grafico("distribuicao_tipo_animal.png")

    plt.figure(figsize=(10, 5))
    ordem_desfechos = df["outcome_type"].value_counts().index
    sns.countplot(data=df, x="outcome_type", order=ordem_desfechos)
    plt.title("Distribuicao por tipo de desfecho")
    plt.xlabel("Tipo de desfecho")
    plt.ylabel("Quantidade de registros")
    plt.xticks(rotation=45, ha="right")
    salvar_grafico("distribuicao_tipo_desfecho.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="age_years", bins=30)
    plt.title("Distribuicao da idade dos animais")
    plt.xlabel("Idade em anos")
    plt.ylabel("Quantidade de registros")
    salvar_grafico("distribuicao_idade.png")
```

Os gráficos gerados foram salvos na pasta `outputs/graficos`, e a tabela com as estatísticas gerais foi salva em `outputs/tabelas/estatisticas_gerais_idade.csv`. Essa etapa cria uma visão inicial da base e prepara o projeto para responder às perguntas investigativas nas próximas partes.

## Parte 4 - Pergunta 1: Proporção de adoções por espécie

Nesta etapa, a análise busca responder à primeira pergunta investigativa: qual espécie apresenta maior proporção de adoções entre seus respectivos desfechos?

A hipótese associada a essa pergunta é que a proporção de adoções difere entre cães e gatos, indicando que o tipo de animal pode estar associado ao resultado de saída do abrigo.

Para responder a essa pergunta, foi calculado o total de registros por espécie, o total de registros cujo desfecho foi `Adoption` e a proporção de adoções dentro de cada tipo de animal. A análise foi feita com proporções, e não apenas com quantidades absolutas, porque cada espécie possui um número diferente de registros na base.

```python
def analisar_adocao_por_especie(df):
    print("\n" + "=" * 80)
    print("4. PERGUNTA 1: PROPORCAO DE ADOCOES POR ESPECIE")
    print("=" * 80)

    tabela = (
        df.groupby("animal_type")
        .agg(
            total_registros=("animal_type", "size"),
            total_adocoes=("outcome_type", lambda coluna: (coluna == "Adoption").sum()),
        )
        .reset_index()
    )
    tabela["proporcao_adocao"] = (
        tabela["total_adocoes"] / tabela["total_registros"]
    )
    tabela["percentual_adocao"] = tabela["proporcao_adocao"] * 100
    tabela = tabela.sort_values("percentual_adocao", ascending=False)

    print("\nProporcao de adocoes por especie:")
    print(tabela)

    tabela.to_csv(
        OUTPUT_TABLES_DIR / "proporcao_adocao_por_especie.csv", index=False
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=tabela, x="animal_type", y="percentual_adocao")
    plt.title("Proporcao de adocoes por especie")
    plt.xlabel("Tipo de animal")
    plt.ylabel("Adocoes (%)")
    salvar_grafico("proporcao_adocao_por_especie.png")

    comparacao_caes_gatos = tabela[tabela["animal_type"].isin(["Dog", "Cat"])]

    print("\nComparacao entre caes e gatos:")
    print(comparacao_caes_gatos)

    return tabela
```

Os resultados obtidos foram:

| Tipo de animal | Total de registros | Total de adoções | Percentual de adoção |
|---|---:|---:|---:|
| Dog | 44.233 | 20.051 | 45,33% |
| Cat | 29.411 | 12.729 | 43,28% |
| Bird | 333 | 114 | 34,23% |
| Livestock | 9 | 2 | 22,22% |
| Other | 4.242 | 212 | 5,00% |

A espécie com maior proporção de adoções foi `Dog`, com aproximadamente 45,33% dos registros resultando em adoção. Em seguida aparecem os gatos, com aproximadamente 43,28%. Embora cães também tenham maior quantidade absoluta de adoções, a comparação proporcional mostra que a diferença permanece mesmo considerando o total de registros de cada espécie.

Considerando especificamente cães e gatos, a hipótese foi sustentada pelos dados, pois as proporções de adoção não são iguais. A diferença observada foi de aproximadamente 2,05 pontos percentuais, com cães apresentando uma proporção de adoção ligeiramente maior do que gatos. Portanto, nesta base, o tipo de animal parece estar associado ao resultado de saída do abrigo quando analisamos o desfecho de adoção.

O gráfico `proporcao_adocao_por_especie.png`, salvo em `outputs/graficos`, mostra visualmente essa comparação entre os percentuais de adoção por espécie. A tabela completa foi salva em `outputs/tabelas/proporcao_adocao_por_especie.csv`.

## Parte 5 - Pergunta 2: Idade de adotados vs transferidos

Nesta etapa, a análise busca responder à segunda pergunta investigativa: a idade dos animais adotados difere da idade dos animais transferidos?

A hipótese associada a essa pergunta é que os animais adotados apresentam uma distribuição de idade diferente dos animais transferidos, com possíveis diferenças na mediana e na concentração de animais mais jovens.

Para fazer essa comparação, foram filtrados apenas os registros com `outcome_type` igual a `Adoption` ou `Transfer`. Em seguida, foram calculadas medidas de centralização, posição e dispersão da idade em anos para cada grupo.

```python
def calcular_moda(serie):
    moda = serie.mode()

    if moda.empty:
        return pd.NA

    return moda.iloc[0]


def analisar_idade_adotados_transferidos(df):
    print("\n" + "=" * 80)
    print("5. PERGUNTA 2: IDADE DE ADOTADOS VS TRANSFERIDOS")
    print("=" * 80)

    df_comparacao = df[df["outcome_type"].isin(["Adoption", "Transfer"])].copy()

    estatisticas = (
        df_comparacao.groupby("outcome_type")["age_years"]
        .agg(
            quantidade="count",
            media="mean",
            mediana="median",
            moda=calcular_moda,
            q1=lambda coluna: coluna.quantile(0.25),
            q3=lambda coluna: coluna.quantile(0.75),
            variancia="var",
            desvio_padrao="std",
            minimo="min",
            maximo="max",
        )
        .reset_index()
    )
    estatisticas["amplitude"] = estatisticas["maximo"] - estatisticas["minimo"]

    print("\nEstatisticas de idade para Adoption e Transfer:")
    print(estatisticas)

    estatisticas.to_csv(
        OUTPUT_TABLES_DIR / "idade_adotados_vs_transferidos.csv", index=False
    )
```

Os resultados obtidos foram:

| Desfecho | Quantidade | Média | Mediana | Moda | Q1 | Q3 | Variância | Desvio padrão | Amplitude |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adoption | 33.108 | 1,67 | 0,82 | 0,16 | 0,16 | 2,00 | 5,75 | 2,40 | 18,00 |
| Transfer | 23.493 | 1,60 | 0,74 | 1,00 | 0,08 | 2,00 | 6,17 | 2,48 | 25,00 |

Os animais adotados apresentaram idade média de aproximadamente 1,67 ano, enquanto os transferidos apresentaram média de aproximadamente 1,60 ano. A mediana também foi ligeiramente maior entre os adotados: 0,82 ano contra 0,74 ano nos transferidos. Isso indica que, na base analisada, os animais adotados não são necessariamente mais jovens do que os transferidos; pelo contrário, aparecem com valores centrais um pouco maiores.

Por outro lado, os animais transferidos apresentaram maior variabilidade de idade. A variância foi de 6,17 no grupo `Transfer`, contra 5,75 no grupo `Adoption`. O desvio padrão também foi maior entre os transferidos, com 2,48 anos contra 2,40 anos. A amplitude reforça essa diferença: os transferidos variam de 0 a 25 anos, enquanto os adotados variam de 0 a 18 anos.

Também foram gerados gráficos para visualizar melhor a comparação entre os grupos:

```python
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_comparacao, x="outcome_type", y="age_years")
    plt.title("Idade dos animais: Adoption vs Transfer")
    plt.xlabel("Tipo de desfecho")
    plt.ylabel("Idade em anos")
    salvar_grafico("boxplot_idade_adoption_transfer.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df_comparacao,
        x="age_years",
        hue="outcome_type",
        bins=30,
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.title("Distribuicao da idade: Adoption vs Transfer")
    plt.xlabel("Idade em anos")
    plt.ylabel("Densidade")
    salvar_grafico("histograma_idade_adoption_transfer.png")
```

A hipótese foi parcialmente sustentada. Os dados mostram que as distribuições de idade de animais adotados e transferidos não são idênticas, pois há diferenças na média, mediana, moda, variância, desvio padrão e amplitude. Porém, a parte da hipótese que sugeria maior concentração de animais mais jovens entre os adotados não foi confirmada pelos valores centrais, já que a mediana dos adotados foi ligeiramente maior do que a dos transferidos.

Portanto, a idade dos animais adotados difere da idade dos animais transferidos, mas essa diferença é moderada e deve ser interpretada com cuidado. O grupo `Transfer` apresentou maior dispersão e maior presença de idades extremas, enquanto o grupo `Adoption` apresentou valores centrais um pouco mais altos.

Os gráficos `boxplot_idade_adoption_transfer.png` e `histograma_idade_adoption_transfer.png` foram salvos em `outputs/graficos`, e a tabela completa foi salva em `outputs/tabelas/idade_adotados_vs_transferidos.csv`.

## Parte 6 - Pergunta 3: Variabilidade de idade por desfecho

Nesta etapa, a análise busca responder à terceira pergunta investigativa: quais grupos de desfecho apresentam maior variabilidade de idade e maior presença de valores extremos?

A hipótese associada a essa pergunta é que a variabilidade da idade não é igual entre os grupos de desfecho, sendo possível que alguns apresentem maior dispersão e maior frequência de animais com idades extremas.

Para responder a essa pergunta, a base foi agrupada por `outcome_type`. Em seguida, foram calculadas medidas de dispersão da idade, como variância, desvio padrão, amplitude e intervalo interquartil. Também foi calculada a quantidade de outliers em cada grupo usando o critério de 1,5 vezes o intervalo interquartil.

```python
def contar_outliers(serie):
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    return ((serie < limite_inferior) | (serie > limite_superior)).sum()


def analisar_variabilidade_por_desfecho(df):
    print("\n" + "=" * 80)
    print("6. PERGUNTA 3: VARIABILIDADE DE IDADE POR DESFECHO")
    print("=" * 80)

    estatisticas = (
        df.groupby("outcome_type")["age_years"]
        .agg(
            quantidade="count",
            media="mean",
            mediana="median",
            q1=lambda coluna: coluna.quantile(0.25),
            q3=lambda coluna: coluna.quantile(0.75),
            variancia="var",
            desvio_padrao="std",
            minimo="min",
            maximo="max",
            outliers=contar_outliers,
        )
        .reset_index()
    )
    estatisticas["amplitude"] = estatisticas["maximo"] - estatisticas["minimo"]
    estatisticas["iqr"] = estatisticas["q3"] - estatisticas["q1"]
    estatisticas["percentual_outliers"] = (
        estatisticas["outliers"] / estatisticas["quantidade"] * 100
    )
    estatisticas = estatisticas.sort_values("desvio_padrao", ascending=False)
```

Os resultados obtidos foram:

| Desfecho | Quantidade | Mediana | Variância | Desvio padrão | Amplitude | IQR | Outliers |
|---|---:|---:|---:|---:|---:|---:|---:|
| Return to Owner | 14.354 | 3,00 | 12,70 | 3,56 | 22,00 | 5,00 | 345 |
| Euthanasia | 6.075 | 1,00 | 11,20 | 3,35 | 22,00 | 2,34 | 673 |
| Rto-Adopt | 150 | 2,00 | 10,34 | 3,22 | 15,84 | 3,75 | 8 |
| Missing | 46 | 0,45 | 8,31 | 2,88 | 14,96 | 1,50 | 4 |
| Died | 680 | 0,16 | 7,91 | 2,81 | 16,00 | 0,92 | 103 |
| Transfer | 23.493 | 0,74 | 6,17 | 2,48 | 25,00 | 1,92 | 2.405 |
| Adoption | 33.108 | 0,82 | 5,75 | 2,40 | 18,00 | 1,84 | 3.558 |
| Disposal | 306 | 1,00 | 1,30 | 1,14 | 12,00 | 0,00 | 137 |
| Relocate | 16 | 0,79 | 0,36 | 0,60 | 1,92 | 0,51 | 3 |

O grupo com maior variabilidade segundo o desvio padrão foi `Return to Owner`, com desvio padrão de 3,56 anos e variância de 12,70. Esse mesmo grupo também apresentou o maior intervalo interquartil, com IQR de 5 anos, indicando uma distribuição mais espalhada entre o primeiro e o terceiro quartil.

O grupo `Transfer` apresentou a maior amplitude, com idades variando de 0 a 25 anos. Isso mostra que, embora seu desvio padrão não seja o maior, esse grupo contém a maior diferença entre idade mínima e máxima.

Em relação à presença de valores extremos, o maior número absoluto de outliers apareceu em `Adoption`, com 3.558 registros identificados como extremos pelo critério de 1,5 vezes o IQR. Em seguida aparece `Transfer`, com 2.405 outliers. Esses números absolutos são influenciados pelo fato de esses grupos também concentrarem grande quantidade de registros. Já proporcionalmente, `Disposal` apresentou percentual elevado de outliers, mas esse resultado deve ser interpretado com cuidado porque esse desfecho possui uma quantidade bem menor de registros.

Também foram gerados gráficos para visualizar a dispersão:

```python
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=df, x="outcome_type", y="age_years", order=ordem_desfechos)
    plt.title("Variabilidade da idade por tipo de desfecho")
    plt.xlabel("Tipo de desfecho")
    plt.ylabel("Idade em anos")
    plt.xticks(rotation=45, ha="right")
    salvar_grafico("boxplot_idade_por_desfecho.png")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=estatisticas, x="outcome_type", y="desvio_padrao")
    plt.title("Desvio padrao da idade por tipo de desfecho")
    plt.xlabel("Tipo de desfecho")
    plt.ylabel("Desvio padrao da idade em anos")
    plt.xticks(rotation=45, ha="right")
    salvar_grafico("desvio_padrao_idade_por_desfecho.png")
```

A hipótese foi sustentada pelos dados. A variabilidade da idade não é igual entre os grupos de desfecho. Alguns grupos, como `Return to Owner`, `Euthanasia` e `Rto-Adopt`, apresentaram maior dispersão segundo o desvio padrão, enquanto `Transfer` apresentou a maior amplitude. Além disso, os grupos `Adoption` e `Transfer` concentraram grande quantidade absoluta de outliers.

Portanto, os desfechos apresentam perfis etários diferentes. `Return to Owner` se destaca como o grupo com maior dispersão geral da idade, enquanto `Transfer` se destaca pela maior distância entre idade mínima e máxima. Os gráficos `boxplot_idade_por_desfecho.png` e `desvio_padrao_idade_por_desfecho.png` foram salvos em `outputs/graficos`, e a tabela completa foi salva em `outputs/tabelas/variabilidade_idade_por_desfecho.csv`.

## Parte 7 - Correlação, padronização e conclusão


Depois de implementar, o relatório deverá apresentar a correlação entre idade e ocorrência de adoção, a padronização de variáveis numéricas em escalas diferentes e a conclusão geral validando ou rejeitando as hipóteses.
