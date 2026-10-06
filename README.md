# Pipeline de treino e avaliação com MLflow: California Housing

Projeto da atividade da Aula 08. O código de regressão sobre o dataset **California Housing** foi reorganizado em um pipeline modular de **carga de dados, treino, validação e teste**, com rastreamento completo no **MLflow** (parâmetros, métricas, datasets, gráficos, logs de execução e modelo).

## Dataset

California Housing, disponível no scikit-learn (`fetch_california_housing`). São 20.640 regiões censitárias da Califórnia (censo de 1990), com 8 variáveis (renda mediana, idade das casas, média de cômodos, população, latitude, longitude etc.). O alvo é o valor mediano das casas, em centenas de milhares de dólares.

Fonte original: Pace, R. K.; Barry, R. *Sparse Spatial Autoregressions*. Statistics & Probability Letters, v. 33, n. 3, p. 291 a 297, 1997.

## Estrutura

```
mlflow-california-housing/
├── configs/                          # um YAML por experimento
│   ├── exp01_baseline_linear.yaml
│   ├── exp02_random_forest.yaml
│   └── exp03_random_forest_features.yaml
├── src/
│   ├── config.py      # leitura do YAML e sobrescrita via CLI
│   ├── data.py        # etapa 1: carga, features derivadas e split treino/val/teste
│   ├── models.py      # etapa 2: construção do modelo a partir da config
│   ├── evaluate.py    # etapa 3: métricas, gráficos e importância de features
│   ├── train.py       # orquestrador do pipeline com MLflow
│   └── compare.py     # tabela comparativa dos runs
├── scripts/run_all.sh # roda todos os experimentos
├── MLproject          # permite executar com `mlflow run`
├── python_env.yaml
└── requirements.txt
```

## Fluxo do pipeline

Cada execução abre um run no MLflow e passa pelas etapas abaixo. Cada etapa registra sua duração (`duration_<etapa>_s`) e escreve no log de execução.

1. **load_data**: baixa o dataset, aplica (ou não) engenharia de features e divide em 60% treino, 20% validação e 20% teste. Os três conjuntos são registrados com `mlflow.log_input`.
2. **cross_validation** (opcional): validação cruzada k-fold apenas no treino.
3. **train**: ajuste do modelo e métricas de treino.
4. **validation**: métricas no conjunto de validação (usado para comparar experimentos).
5. **test**: avaliação final no conjunto de teste.
6. **artifacts**: gráficos (previsto vs real, resíduos, importância das features), tabela de importância, config usada, `pipeline.log` e o modelo com assinatura.

Métricas registradas: RMSE, MAE e R² em treino, validação e teste.

## Experimentos e perguntas

| Experimento | Pergunta | Configuração |
|---|---|---|
| exp01_baseline_linear | Qual o desempenho de um modelo linear simples como linha de base? | Ridge, 8 features originais |
| exp02_random_forest | Um modelo não linear reduz o erro em relação à linha de base, com as mesmas features? | Random Forest, 8 features originais |
| exp03_random_forest_features | Features derivadas melhoram o Random Forest do exp02? | Random Forest, 13 features (razões e distância a Los Angeles e San Francisco) |

A pergunta e a hipótese de cada experimento ficam registradas como tags do run (`pergunta`, `hipotese`).

## Resultados obtidos

| Run | CV RMSE (treino) | RMSE val | RMSE teste | MAE teste | R² teste |
|---|---|---|---|---|---|
| exp01_baseline_linear | 1,164 | 0,728 | 0,749 | 0,533 | 0,571 |
| exp02_random_forest | 0,519 | 0,508 | 0,507 | 0,329 | 0,803 |
| exp03_random_forest_features | 0,496 | 0,480 | 0,477 | 0,308 | 0,826 |

Respostas:

* **exp01**: o modelo linear explica cerca de 57% da variância. O desvio alto na validação cruzada (±0,90) mostra que ele é sensível a valores extremos (algumas regiões têm média de ocupação muito alta).
* **exp02**: sim. O Random Forest reduz o RMSE de teste em cerca de 32% e sobe o R² para 0,80.
* **exp03**: sim, de forma moderada. As features derivadas reduzem o RMSE de teste em cerca de 6% em relação ao exp02. Veja a importância das features no MLflow para conferir o peso das distâncias.

*(Valores com `seed: 42`; podem variar levemente entre versões de bibliotecas.)*

## Como executar

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# um experimento
python -m src.train --config configs/exp01_baseline_linear.yaml

# todos os experimentos + tabela comparativa
bash scripts/run_all.sh

# comparação (salva em outputs/comparacao_runs.csv)
python -m src.compare
```

### Parametrização pela linha de comando

Qualquer chave do YAML pode ser sobrescrita sem editar o arquivo:

```bash
python -m src.train --config configs/exp02_random_forest.yaml \
  --set run_name=rf_400_arvores model.params.n_estimators=400 model.params.max_depth=null
```

Modelos disponíveis em `model.type`: `linear_regression`, `ridge`, `random_forest`, `hist_gradient_boosting`.

### Interface do MLflow

O rastreamento usa o banco local `sqlite:///mlflow.db` (pode ser trocado pela variável `MLFLOW_TRACKING_URI`).

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abra http://127.0.0.1:5000, entre no experimento `california-housing`, selecione os runs e use **Compare** para ver métricas lado a lado.

### Executando como MLflow Project

```bash
mlflow run . -P config=configs/exp03_random_forest_features.yaml --env-manager local
```

## Referências

* MLflow Tracking: https://mlflow.org/docs/latest/ml/tracking
* MLflow Dataset Tracking (`mlflow.log_input`): https://mlflow.org/docs/latest/ml/dataset/
* `fetch_california_housing` (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
* Pace, R. K.; Barry, R. (1997). Sparse Spatial Autoregressions. Statistics & Probability Letters, 33(3), 291 a 297.
