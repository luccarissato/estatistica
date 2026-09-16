from pathlib import Path
import os
import re

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = BASE_DIR / "data" / "aac_shelter_outcomes.csv"
OUTPUT_GRAPHS_DIR = BASE_DIR / "outputs" / "graficos"
OUTPUT_TABLES_DIR = BASE_DIR / "outputs" / "tabelas"


def carregar_base():
    return pd.read_csv(DATA_PATH)


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


def converter_idade_para_dias(idade):
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

    df_processado = df_processado.drop_duplicates().copy()

    df_processado["date_of_birth"] = pd.to_datetime(
        df_processado["date_of_birth"], errors="coerce"
    )
    df_processado["datetime"] = pd.to_datetime(
        df_processado["datetime"], errors="coerce"
    )
    df_processado["monthyear"] = pd.to_datetime(
        df_processado["monthyear"], errors="coerce"
    )

    df_processado["age_days"] = df_processado["age_upon_outcome"].apply(
        converter_idade_para_dias
    )
    df_processado["age_days"] = pd.to_numeric(
        df_processado["age_days"], errors="coerce"
    )
    df_processado["age_years"] = df_processado["age_days"] / 365

    df_processado["outcome_year"] = df_processado["datetime"].dt.year

    df_processado = df_processado.dropna(subset=["age_days", "outcome_type"]).copy()

    print("\nValores ausentes depois do tratamento essencial:")
    print(df_processado.isna().sum())

    print("\nTipos de dados depois das conversoes:")
    print(df_processado.dtypes)

    print("\nQuantidade de linhas e colunas depois do pre-processamento:")
    print(df_processado.shape)

    return df_processado


def salvar_grafico(nome_arquivo):
    plt.tight_layout()
    plt.savefig(OUTPUT_GRAPHS_DIR / nome_arquivo, dpi=300)
    plt.close()


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

    print("\nArquivos gerados:")
    print(OUTPUT_TABLES_DIR / "estatisticas_gerais_idade.csv")
    print(OUTPUT_GRAPHS_DIR / "distribuicao_tipo_animal.png")
    print(OUTPUT_GRAPHS_DIR / "distribuicao_tipo_desfecho.png")
    print(OUTPUT_GRAPHS_DIR / "distribuicao_idade.png")

    return estatisticas


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


def main():
    df = carregar_base()
    entender_base(df)
    df_processado = preprocessar_dados(df)
    analisar_estatisticas_gerais(df_processado)
    analisar_adocao_por_especie(df_processado)


if __name__ == "__main__":
    main()
