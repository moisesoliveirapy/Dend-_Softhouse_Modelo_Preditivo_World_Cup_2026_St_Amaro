from pathlib import Path
import warnings

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_TREINO = BASE_DIR / "datasets" / "train.csv"
ARQUIVO_TESTE = BASE_DIR / "datasets" / "test.csv"
ARQUIVO_RANKING = BASE_DIR / "ranking_probabilidades.csv"

COLUNA_ALVO = "winner"
COLUNAS_IDENTIFICACAO = ["team_name", "country_code"]
COLUNAS_CATEGORICAS = ["confederation"]

VALORES_C = [0.1, 1.0, 10.0]
N_FOLDS = 5
SEMENTE_ALEATORIA = 42


def carregar_dados(caminho_treino, caminho_teste):
    df_treino = pd.read_csv(caminho_treino)
    df_teste = pd.read_csv(caminho_teste)

    print("=" * 70)
    print("1. CARREGAMENTO DOS DADOS")
    print("=" * 70)
    print(f"Treino: {df_treino.shape[0]} registros e {df_treino.shape[1]} colunas")
    print(f"Teste : {df_teste.shape[0]} registros e {df_teste.shape[1]} colunas")

    return df_treino, df_teste


def explorar_dataset(df_treino):
    print("\n" + "=" * 70)
    print("2. EXPLORACAO DO DATASET")
    print("=" * 70)

    colunas_categoricas = df_treino.select_dtypes(include=["object"]).columns.tolist()
    colunas_numericas = df_treino.select_dtypes(include=["number"]).columns.tolist()

    print(f"Variavel-alvo           : {COLUNA_ALVO}")
    print(f"Colunas categoricas     : {colunas_categoricas}")
    print(f"Quantidade numericas    : {len(colunas_numericas)}")
    print(f"Valores ausentes        : {int(df_treino.isnull().sum().sum())}")
    print(f"Registros duplicados    : {int(df_treino.duplicated().sum())}")

    print("\nDistribuicao da variavel-alvo (winner):")
    print(df_treino[COLUNA_ALVO].value_counts().to_string())


def selecionar_caracteristicas(df_treino):
    print("\n" + "=" * 70)
    print("3. ESCOLHA DAS CARACTERISTICAS")
    print("=" * 70)

    caracteristicas = [
        coluna for coluna in df_treino.columns
        if coluna not in COLUNAS_IDENTIFICACAO and coluna != COLUNA_ALVO
    ]
    colunas_categoricas = [
        coluna for coluna in COLUNAS_CATEGORICAS
        if coluna in caracteristicas
    ]
    colunas_numericas = [
        coluna for coluna in caracteristicas
        if coluna not in colunas_categoricas
    ]

    print(f"Colunas removidas do treino: {COLUNAS_IDENTIFICACAO}")
    print("Motivo: sao identificadores usados apenas para exibir o ranking final.")
    print(f"\nColuna categorica com OneHotEncoder: {colunas_categoricas}")
    print(f"Colunas numericas com StandardScaler: {len(colunas_numericas)}")
    print(f"Total de caracteristicas usadas     : {len(caracteristicas)}")

    return caracteristicas, colunas_numericas, colunas_categoricas


def criar_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def criar_pipeline(valor_c, colunas_numericas, colunas_categoricas):
    preprocessamento_numerico = Pipeline(steps=[
        ("imputador", SimpleImputer(strategy="median")),
        ("padronizador", StandardScaler()),
    ])

    preprocessamento_categorico = Pipeline(steps=[
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("one_hot", criar_one_hot_encoder()),
    ])

    transformador = ColumnTransformer(
        transformers=[
            ("numericas", preprocessamento_numerico, colunas_numericas),
            ("categoricas", preprocessamento_categorico, colunas_categoricas),
        ],
        remainder="drop",
    )

    return Pipeline(steps=[
        ("preprocessamento", transformador),
        ("modelo", LogisticRegression(
            C=valor_c,
            solver="lbfgs",
            max_iter=1000,
            random_state=SEMENTE_ALEATORIA,
        )),
    ])


def comparar_hiperparametros(X_treino, y_treino, colunas_numericas, colunas_categoricas):
    print("\n" + "=" * 70)
    print("5. VALIDACAO CRUZADA NO TRAIN.CSV")
    print("=" * 70)
    print(f"Registros usados na validacao cruzada: {len(X_treino)}")
    print(f"Folds: {N_FOLDS}")

    validacao = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEMENTE_ALEATORIA,
    )
    metricas = {
        "acuracia": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    resultados = []
    print(f"\n{'C':>6}{'Acuracia':>14}{'F1':>14}{'ROC AUC':>14}")
    print("-" * 48)

    for valor_c in VALORES_C:
        pipeline = criar_pipeline(valor_c, colunas_numericas, colunas_categoricas)
        scores = cross_validate(
            pipeline,
            X_treino,
            y_treino,
            cv=validacao,
            scoring=metricas,
        )

        resultado = {
            "C": valor_c,
            "acuracia_media": scores["test_acuracia"].mean(),
            "f1_medio": scores["test_f1"].mean(),
            "roc_auc_medio": scores["test_roc_auc"].mean(),
        }
        resultados.append(resultado)

        print(
            f"{valor_c:>6.1f}"
            f"{resultado['acuracia_media']:>14.4f}"
            f"{resultado['f1_medio']:>14.4f}"
            f"{resultado['roc_auc_medio']:>14.4f}"
        )

    df_resultados = pd.DataFrame(resultados)
    melhor = df_resultados.sort_values(
        by=["f1_medio", "roc_auc_medio", "acuracia_media"],
        ascending=False,
    ).iloc[0]

    print(f"\nMelhor configuracao: C = {melhor['C']:.1f}")
    return float(melhor["C"])


def treinar_modelo_final(valor_c, X_treino, y_treino, colunas_numericas, colunas_categoricas):
    print("\n" + "=" * 70)
    print("6. TREINAMENTO FINAL")
    print("=" * 70)
    print(f"Modelo final treinado com 100% do train.csv: {len(X_treino)} registros")

    modelo_final = criar_pipeline(valor_c, colunas_numericas, colunas_categoricas)
    modelo_final.fit(X_treino, y_treino)

    return modelo_final


def gerar_ranking_probabilidades(modelo_final, X_teste, df_teste_original):
    print("\n" + "=" * 70)
    print("7. PROBABILIDADES FINAIS NO TEST.CSV")
    print("=" * 70)
    print(f"Registros usados para probabilidades finais: {len(X_teste)}")

    indice_classe_positiva = list(modelo_final.classes_).index(1)
    probabilidades = modelo_final.predict_proba(X_teste)[:, indice_classe_positiva]

    df_resultado = df_teste_original[COLUNAS_IDENTIFICACAO].copy()
    df_resultado["probabilidade"] = probabilidades

    ranking = (
        df_resultado
        .groupby(COLUNAS_IDENTIFICACAO, as_index=False)["probabilidade"]
        .mean()
        .sort_values("probabilidade", ascending=False)
        .reset_index(drop=True)
    )
    ranking.insert(0, "posicao", range(1, len(ranking) + 1))
    ranking["probabilidade_pct"] = (ranking["probabilidade"] * 100).round(2)

    ranking.to_csv(ARQUIVO_RANKING, index=False)

    print("TOP 10 SELECOES FAVORITAS:\n")
    print(f"{'Pos':<5}{'Pais':<18}{'Codigo':<8}{'Probabilidade':>14}")
    print("-" * 45)
    for _, linha in ranking.head(10).iterrows():
        print(
            f"{linha['posicao']:<5}"
            f"{linha['team_name']:<18}"
            f"{linha['country_code']:<8}"
            f"{linha['probabilidade_pct']:>12.2f}%"
        )

    print(f"\nRanking completo salvo em: {ARQUIVO_RANKING}")
    return ranking


def main():
    df_treino, df_teste = carregar_dados(ARQUIVO_TREINO, ARQUIVO_TESTE)
    df_teste_original = df_teste.copy()

    explorar_dataset(df_treino)
    caracteristicas, colunas_numericas, colunas_categoricas = selecionar_caracteristicas(df_treino)

    X_treino = df_treino[caracteristicas].copy()
    y_treino = df_treino[COLUNA_ALVO].copy()
    X_teste = df_teste[caracteristicas].copy()

    melhor_c = comparar_hiperparametros(
        X_treino,
        y_treino,
        colunas_numericas,
        colunas_categoricas,
    )
    modelo_final = treinar_modelo_final(
        melhor_c,
        X_treino,
        y_treino,
        colunas_numericas,
        colunas_categoricas,
    )
    gerar_ranking_probabilidades(modelo_final, X_teste, df_teste_original)


if __name__ == "__main__":
    main()
