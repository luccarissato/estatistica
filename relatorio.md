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

## Parte 4 - Medidas de posição



Depois de implementar, o relatório deverá interpretar os quartis e relacioná-los ao comportamento geral da base.

## Parte 5 - Medidas de dispersão



Depois de implementar, o relatório deverá explicar se existe grande variação na idade dos animais e o que isso significa para a análise.

## Parte 6 - Correlação



Depois de implementar, o relatório deverá explicar se existe relação estatística relevante entre as variáveis comparadas.

## Parte 7 - Padronização, visualização e conclusões


Depois de implementar, o relatório deverá apresentar os gráficos, interpretar os resultados e responder diretamente às perguntas definidas na introdução.
