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


def entender_base(df):
    # Parte 1 - Preparacao e entendimento da base.
    # Esta etapa confirma o tamanho real do dataset, as colunas disponiveis,
    # os tipos de dados e as categorias mais importantes para as perguntas
    # investigativas do projeto.

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


def preprocessar_dados(df):
    # Parte 2 - Pre-processamento dos dados.
    # Esta etapa prepara a base para responder as perguntas investigativas,
    # tratando valores ausentes, duplicados e tipos de dados.

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


def main():
    df = carregar_base()
    entender_base(df)
    df_processado = preprocessar_dados(df)


if __name__ == "__main__":
    main()
