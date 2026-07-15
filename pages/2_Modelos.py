from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Modelos | Prestamos",
    page_icon=":brain:",
    layout="wide",
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"
MODELS_DIR = PROJECT_DIR / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"

MODEL_FILES = {
    "Logistic Regression": "logistic.pkl",
    "Random Forest": "randomforest.pkl",
    "Support Vector Machine": "svm.pkl",
    "KNN": "knn.pkl",
    "Gaussian Naive Bayes": "naivebayes.pkl",
}


def apply_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bank-navy: #0f172a;
                --bank-blue: #1d4ed8;
                --bank-blue-soft: #dbeafe;
                --bank-slate: #475569;
                --bank-border: #dbe3ef;
                --bank-bg: #f5f7fb;
                --bank-card: #ffffff;
            }

            .stApp {
                background: var(--bank-bg);
            }

            .main .block-container {
                padding-top: 1.25rem;
                padding-bottom: 3rem;
                max-width: 1220px;
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f172a 0%, #172554 100%);
                border-right: 1px solid rgba(255,255,255,.08);
            }

            section[data-testid="stSidebar"] h1,
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3,
            section[data-testid="stSidebar"] p,
            section[data-testid="stSidebar"] span,
            section[data-testid="stSidebar"] label {
                color: #f8fafc;
            }

            .models-hero {
                background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
                border: 1px solid #bfdbfe;
                border-left: 6px solid var(--bank-blue);
                border-radius: 8px;
                padding: 1.65rem 1.85rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 18px 42px rgba(15, 23, 42, .08);
            }

            .models-hero h1 {
                color: var(--bank-navy);
                margin: 0 0 .35rem 0;
            }

            .models-hero p {
                color: var(--bank-slate);
                margin: 0;
                font-size: 1.02rem;
            }

            .best-model-card {
                border: 1px solid #bfdbfe;
                border-radius: 8px;
                padding: 1.2rem 1.35rem;
                background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
                box-shadow: 0 12px 30px rgba(37, 99, 235, 0.12);
                margin-bottom: 1rem;
            }

            .best-model-label {
                color: #1d4ed8;
                font-size: .86rem;
                font-weight: 700;
                text-transform: uppercase;
                margin-bottom: .3rem;
            }

            .best-model-name {
                color: #0f172a;
                font-size: 1.65rem;
                font-weight: 800;
                line-height: 1.1;
            }

            .best-model-note {
                color: #475569;
                margin-top: .35rem;
            }

            div[data-testid="stPlotlyChart"] {
                border: 1px solid var(--bank-border);
                border-radius: 8px;
                padding: .5rem;
                background: var(--bank-card);
                box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
            }

            div[data-testid="stDataFrame"] {
                border: 1px solid var(--bank-border);
                border-radius: 8px;
                background: #ffffff;
                box-shadow: 0 8px 22px rgba(15, 23, 42, .04);
            }

            div[data-testid="stSelectbox"] > div {
                background: #ffffff;
                border-radius: 8px;
            }

            h1, h2, h3 {
                color: var(--bank-navy);
                letter-spacing: 0;
            }

            div[data-testid="stAlert"] {
                border-radius: 8px;
                border: 1px solid var(--bank-border);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Carga el primer CSV disponible del directorio dataset."""
    preferred_names = [
        "loan_prediction.csv",
        "LoanPrediction.csv",
        "train.csv",
        "train_u6lujuX_CVtuZ9i.csv",
        "loan_data.csv",
    ]

    for file_name in preferred_names:
        dataset_path = DATASET_DIR / file_name
        if dataset_path.exists():
            return pd.read_csv(dataset_path)

    csv_files = sorted(DATASET_DIR.glob("*.csv"))
    if csv_files:
        return pd.read_csv(csv_files[0])

    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Carga modelos, scaler, encoder y metadata guardados con Joblib."""
    loaded_models = {}
    missing_files = []

    for model_name, file_name in MODEL_FILES.items():
        file_path = MODELS_DIR / file_name
        if file_path.exists():
            loaded_models[model_name] = joblib.load(file_path)
        else:
            missing_files.append(file_name)

    scaler_path = MODELS_DIR / "scaler.pkl"
    encoder_path = MODELS_DIR / "encoder.pkl"
    columns_path = MODELS_DIR / "columns.pkl"

    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    encoder = joblib.load(encoder_path) if encoder_path.exists() else None
    metadata = joblib.load(columns_path) if columns_path.exists() else None

    for path in [scaler_path, encoder_path, columns_path]:
        if not path.exists():
            missing_files.append(path.name)

    return loaded_models, scaler, encoder, metadata, missing_files


@st.cache_data(show_spinner=False)
def load_saved_metrics():
    """Carga metrics.json para mostrar evaluacion sin recalcular ni entrenar."""
    if not METRICS_PATH.exists():
        return None

    with open(METRICS_PATH, "r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def normalize_target(y_raw: pd.Series, target_mapping: dict) -> pd.Series:
    """Convierte la variable objetivo a valores binarios 0/1."""
    mapped = y_raw.map(target_mapping)
    if mapped.isnull().any():
        fallback_mapping = {
            "y": 1,
            "yes": 1,
            "approved": 1,
            "aprobado": 1,
            "1": 1,
            "true": 1,
            "n": 0,
            "no": 0,
            "rejected": 0,
            "rechazado": 0,
            "0": 0,
            "false": 0,
        }
        mapped = y_raw.astype(str).str.strip().str.lower().map(fallback_mapping)

    if mapped.isnull().any():
        raise ValueError("La variable objetivo contiene valores no reconocidos.")

    return mapped.astype(int)


def prepare_test_data(df: pd.DataFrame, scaler, encoder, metadata):
    """Replica el preprocesamiento usado en entrenamiento y devuelve el set de test."""
    target_column = metadata["target_column"]
    id_columns = metadata.get("id_columns", [])
    numeric_features = metadata.get("numeric_features", [])
    categorical_features = metadata.get("categorical_features", [])
    target_mapping = metadata.get("target_mapping", {"Y": 1, "N": 0, 1: 1, 0: 0})
    imputation_values = metadata.get("imputation_values", {})
    outlier_limits = metadata.get("outlier_limits", {})

    if target_column not in df.columns:
        raise ValueError(f"No se encontro la columna objetivo '{target_column}' en el dataset.")

    work_df = df.copy()
    work_df.columns = work_df.columns.str.strip()

    X = work_df.drop(columns=[target_column] + id_columns, errors="ignore")
    y = normalize_target(work_df[target_column], target_mapping)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_test_clean = X_test.copy()

    for column in numeric_features:
        value = imputation_values.get("numeric", {}).get(column, X_train[column].median())
        X_test_clean[column] = X_test_clean[column].fillna(value)

    for column in categorical_features:
        value = imputation_values.get("categorical", {}).get(column, "Desconocido")
        X_test_clean[column] = X_test_clean[column].fillna(value)

    for column, limits in outlier_limits.items():
        if column in X_test_clean.columns:
            X_test_clean[column] = X_test_clean[column].clip(
                lower=limits["lower"],
                upper=limits["upper"],
            )

    if numeric_features:
        X_test_scaled = scaler.transform(X_test_clean[numeric_features])
    else:
        X_test_scaled = np.empty((len(X_test_clean), 0))

    if categorical_features:
        X_test_encoded = encoder.transform(X_test_clean[categorical_features])
    else:
        X_test_encoded = np.empty((len(X_test_clean), 0))

    X_test_processed = np.hstack([X_test_scaled, X_test_encoded])
    return X_test_processed, y_test


def get_positive_class_scores(model, X_data):
    """Obtiene scores para ROC/AUC desde predict_proba o decision_function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_data)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_data)
    return model.predict(X_data)


def evaluate_models(models: dict, X_test, y_test):
    """Calcula metricas, curvas ROC y matrices de confusion."""
    rows = []
    roc_data = {}
    confusion_data = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_score = get_positive_class_scores(model, X_test)

        rows.append(
            {
                "Modelo": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0),
                "AUC": roc_auc_score(y_test, y_score),
            }
        )

        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_data[model_name] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
        confusion_data[model_name] = confusion_matrix(y_test, y_pred)

    results_df = pd.DataFrame(rows)
    return results_df.sort_values(["F1", "AUC"], ascending=False).reset_index(drop=True), roc_data, confusion_data


def render_best_model(results_df: pd.DataFrame) -> None:
    best_model = results_df.iloc[0]
    st.markdown(
        f"""
        <div class="best-model-card">
            <div class="best-model-label">Mejor modelo automatico</div>
            <div class="best-model-name">{best_model["Modelo"]}</div>
            <div class="best-model-note">
                Seleccionado por F1 y AUC. F1: {best_model["F1"]:.3f} | AUC: {best_model["AUC"]:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_metrics_bar(results_df: pd.DataFrame):
    metrics_df = results_df.melt(
        id_vars="Modelo",
        value_vars=["Accuracy", "Precision", "Recall", "F1", "AUC"],
        var_name="Metrica",
        value_name="Valor",
    )
    fig = px.bar(
        metrics_df,
        x="Modelo",
        y="Valor",
        color="Metrica",
        barmode="group",
        title="Comparacion de metricas por modelo",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        height=460,
        yaxis_range=[0, 1],
        margin=dict(l=20, r=20, t=60, b=60),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#334155"),
    )
    fig.update_yaxes(gridcolor="#e2e8f0")
    fig.update_xaxes(tickangle=20)
    return fig


def plot_roc_curves(results_df: pd.DataFrame, roc_data: dict):
    fig = go.Figure()

    for model_name, values in roc_data.items():
        auc_value = results_df.loc[results_df["Modelo"] == model_name, "AUC"].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=values["fpr"],
                y=values["tpr"],
                mode="lines",
                name=f"{model_name} (AUC={auc_value:.3f})",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Clasificador aleatorio",
            line=dict(color="#94a3b8", dash="dash"),
        )
    )

    fig.update_layout(
        title="Curvas ROC",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=460,
        margin=dict(l=20, r=20, t=60, b=50),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#334155"),
    )
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0")
    return fig


def plot_confusion_matrix(matrix: np.ndarray, model_name: str):
    fig = ff.create_annotated_heatmap(
        z=matrix,
        x=["Pred. Rechazado", "Pred. Aprobado"],
        y=["Real Rechazado", "Real Aprobado"],
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title=f"Matriz de confusion - {model_name}",
        height=380,
        margin=dict(l=20, r=20, t=60, b=40),
        paper_bgcolor="#ffffff",
        font=dict(color="#334155"),
    )
    return fig


def plot_random_forest_importance(model, metadata):
    feature_names = metadata.get("feature_names", [])
    importances = getattr(model, "feature_importances_", None)

    if importances is None or not feature_names:
        return None

    importance_df = (
        pd.DataFrame({"Variable": feature_names, "Importancia": importances})
        .sort_values("Importancia", ascending=False)
        .head(15)
    )

    fig = px.bar(
        importance_df,
        x="Importancia",
        y="Variable",
        orientation="h",
        title="Importancia de variables - Random Forest",
        color="Importancia",
        color_continuous_scale="Teal",
    )
    fig.update_layout(
        height=520,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=60, b=40),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#334155"),
    )
    fig.update_xaxes(gridcolor="#e2e8f0")
    return fig


apply_styles()

st.markdown(
    """
    <div class="models-hero">
        <h1>Comparacion de Modelos</h1>
        <p>Evaluacion de modelos entrenados para prediccion de aprobacion de prestamos.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

models, scaler, encoder, metadata, missing_files = load_artifacts()
saved_metrics = load_saved_metrics()
df = load_dataset()

if missing_files:
    st.warning(
        "Faltan artefactos de entrenamiento en la carpeta models/: "
        + ", ".join(missing_files)
    )

if saved_metrics is not None:
    results_df = pd.DataFrame(saved_metrics.get("metrics", []))
    confusion_data = {
        model_name: np.array(matrix)
        for model_name, matrix in saved_metrics.get("confusion_matrices", {}).items()
    }
    roc_data = saved_metrics.get("roc_curves", {})

    if results_df.empty:
        st.error("metrics.json existe, pero no contiene metricas validas.")
        st.stop()

elif not models or scaler is None or encoder is None or metadata is None:
    st.info(
        "Ejecuta primero el notebook entrenamiento.ipynb para generar los modelos, "
        "el scaler, el encoder, columns.pkl y metrics.json."
    )
    st.stop()

elif df.empty:
    st.info(
        "No se encontro ningun CSV en dataset/. Agrega el Loan Prediction Dataset "
        "para calcular metricas si no existe metrics.json."
    )
    st.stop()

else:
    try:
        X_test_processed, y_test = prepare_test_data(df, scaler, encoder, metadata)
        results_df, roc_data, confusion_data = evaluate_models(models, X_test_processed, y_test)
    except Exception as exc:
        st.error(f"No fue posible cargar los resultados de evaluacion: {exc}")
        st.stop()

render_best_model(results_df)

st.markdown("### Tabla comparativa")
st.dataframe(
    results_df.style.format(
        {
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1": "{:.3f}",
            "AUC": "{:.3f}",
        }
    ).highlight_max(subset=["Accuracy", "Precision", "Recall", "F1", "AUC"], color="#dbeafe"),
    use_container_width=True,
    hide_index=True,
)

st.markdown("### Metricas y curvas")
left_col, right_col = st.columns([1, 1])

with left_col:
    st.plotly_chart(plot_metrics_bar(results_df), use_container_width=True)

with right_col:
    st.plotly_chart(plot_roc_curves(results_df, roc_data), use_container_width=True)

st.markdown("### Matrices de confusion")
selected_model = st.selectbox("Modelo", list(confusion_data.keys()))
st.plotly_chart(
    plot_confusion_matrix(confusion_data[selected_model], selected_model),
    use_container_width=True,
)

st.markdown("### Random Forest")
if "Random Forest" in models:
    importance_fig = plot_random_forest_importance(models["Random Forest"], metadata)
    if importance_fig is not None:
        st.plotly_chart(importance_fig, use_container_width=True)
    else:
        st.info("El modelo Random Forest no contiene importancias de variables disponibles.")
else:
    st.info("No se encontro el archivo randomforest.pkl.")
