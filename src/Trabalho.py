from pathlib import Path

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


def main():
    df = carregar_base()
    entender_base(df)


if __name__ == "__main__":
    main()
