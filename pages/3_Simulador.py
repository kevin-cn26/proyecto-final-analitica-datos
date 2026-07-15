from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Simulador | Prestamos",
    page_icon=":memo:",
    layout="wide",
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"

MODEL_FILES = {
    "Logistic Regression": "logistic.pkl",
    "Random Forest": "randomforest.pkl",
    "Support Vector Machine": "svm.pkl",
    "KNN": "knn.pkl",
    "Gaussian Naive Bayes": "naivebayes.pkl",
}

FORM_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]


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

            .sim-hero {
                background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
                border: 1px solid #bfdbfe;
                border-left: 6px solid var(--bank-blue);
                border-radius: 8px;
                padding: 1.65rem 1.85rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 18px 42px rgba(15, 23, 42, .08);
            }

            .sim-hero h1 {
                color: var(--bank-navy);
                margin: 0 0 .35rem 0;
            }

            .sim-hero p {
                color: var(--bank-slate);
                margin: 0;
                font-size: 1.02rem;
            }

            .decision-card {
                border: 1px solid #bfdbfe;
                border-radius: 8px;
                padding: 1.25rem 1.35rem;
                background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
                box-shadow: 0 12px 30px rgba(37, 99, 235, 0.12);
                margin-bottom: 1rem;
            }

            .decision-card.rejected {
                border-color: #fecaca;
                background: #fef2f2;
                box-shadow: 0 10px 28px rgba(220, 38, 38, 0.08);
            }

            .decision-label {
                color: #475569;
                font-size: .86rem;
                font-weight: 700;
                text-transform: uppercase;
                margin-bottom: .3rem;
            }

            .decision-value {
                color: #0f172a;
                font-size: 1.7rem;
                font-weight: 800;
                line-height: 1.15;
            }

            .decision-note {
                color: #475569;
                margin-top: .4rem;
            }

            div[data-testid="stPlotlyChart"] {
                border: 1px solid var(--bank-border);
                border-radius: 8px;
                padding: .5rem;
                background: var(--bank-card);
                box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
            }

            div[data-testid="stVerticalBlock"] div[data-testid="stNumberInput"],
            div[data-testid="stVerticalBlock"] div[data-testid="stSelectbox"] {
                background: #ffffff;
                border-radius: 8px;
            }

            div[data-testid="stDataFrame"] {
                border: 1px solid var(--bank-border);
                border-radius: 8px;
                background: #ffffff;
                box-shadow: 0 8px 22px rgba(15, 23, 42, .04);
            }

            .stButton > button {
                background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
                color: #ffffff;
                border: 0;
                border-radius: 8px;
                min-height: 3rem;
                font-weight: 750;
                box-shadow: 0 10px 24px rgba(29, 78, 216, .24);
            }

            .stButton > button:hover {
                border: 0;
                color: #ffffff;
                filter: brightness(.98);
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


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Carga modelos y transformadores ya entrenados. No entrena nada."""
    loaded_models = {}
    missing_files = []

    for model_name, file_name in MODEL_FILES.items():
        model_path = MODELS_DIR / file_name
        if model_path.exists():
            loaded_models[model_name] = joblib.load(model_path)
        else:
            missing_files.append(file_name)

    scaler_path = MODELS_DIR / "scaler.pkl"
    encoder_path = MODELS_DIR / "encoder.pkl"
    columns_path = MODELS_DIR / "columns.pkl"

    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    encoder = joblib.load(encoder_path) if encoder_path.exists() else None
    metadata = joblib.load(columns_path) if columns_path.exists() else None

    for artifact_path in [scaler_path, encoder_path, columns_path]:
        if not artifact_path.exists():
            missing_files.append(artifact_path.name)

    return loaded_models, scaler, encoder, metadata, missing_files


def get_probability(model, transformed_record: np.ndarray) -> float:
    """Devuelve probabilidad de aprobacion usando la mejor salida disponible."""
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(transformed_record)[:, 1][0])

    if hasattr(model, "decision_function"):
        score = float(model.decision_function(transformed_record)[0])
        return float(1 / (1 + np.exp(-score)))

    return float(model.predict(transformed_record)[0])


def get_prediction_label(probability: float) -> str:
    return "Aprobado" if probability >= 0.50 else "Rechazado"


def build_input_record(form_values: dict, metadata: dict) -> pd.DataFrame:
    """Construye el registro con las columnas esperadas por el entrenamiento."""
    expected_numeric = metadata.get("numeric_features", [])
    expected_categorical = metadata.get("categorical_features", [])
    expected_columns = expected_numeric + expected_categorical

    record = {column: form_values.get(column, np.nan) for column in expected_columns}
    return pd.DataFrame([record])


def preprocess_record(record: pd.DataFrame, scaler, encoder, metadata: dict) -> np.ndarray:
    """Aplica imputacion, tratamiento de outliers, One Hot Encoding y escalado."""
    numeric_features = metadata.get("numeric_features", [])
    categorical_features = metadata.get("categorical_features", [])
    imputation_values = metadata.get("imputation_values", {})
    outlier_limits = metadata.get("outlier_limits", {})

    processed = record.copy()

    for column in numeric_features:
        default_value = imputation_values.get("numeric", {}).get(column, 0)
        processed[column] = pd.to_numeric(processed[column], errors="coerce").fillna(default_value)

    for column in categorical_features:
        default_value = imputation_values.get("categorical", {}).get(column, "Desconocido")
        processed[column] = processed[column].fillna(default_value).astype(str)

    for column, limits in outlier_limits.items():
        if column in processed.columns:
            processed[column] = processed[column].clip(
                lower=limits["lower"],
                upper=limits["upper"],
            )

    if numeric_features:
        numeric_scaled = scaler.transform(processed[numeric_features])
    else:
        numeric_scaled = np.empty((len(processed), 0))

    if categorical_features:
        categorical_encoded = encoder.transform(processed[categorical_features])
    else:
        categorical_encoded = np.empty((len(processed), 0))

    return np.hstack([numeric_scaled, categorical_encoded])


def render_decision_card(final_decision: str, recommended_model: str, probability: float) -> None:
    css_class = "decision-card" if final_decision == "Aprobado" else "decision-card rejected"
    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="decision-label">Decision final</div>
            <div class="decision-value">{final_decision}</div>
            <div class="decision-note">
                Modelo recomendado: <strong>{recommended_model}</strong> |
                Probabilidad estimada: <strong>{probability:.2%}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_form() -> dict:
    st.markdown("### Datos de la solicitud")

    col_1, col_2, col_3 = st.columns(3)

    with col_1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])

    with col_2:
        self_employed = st.selectbox("Self_Employed", ["No", "Yes"])
        applicant_income = st.number_input("ApplicantIncome", min_value=0.0, value=5000.0, step=100.0)
        coapplicant_income = st.number_input("CoapplicantIncome", min_value=0.0, value=0.0, step=100.0)
        loan_amount = st.number_input("LoanAmount", min_value=0.0, value=150.0, step=10.0)

    with col_3:
        loan_amount_term = st.number_input("Loan_Amount_Term", min_value=0.0, value=360.0, step=12.0)
        credit_history = st.selectbox("Credit_History", [1.0, 0.0], format_func=lambda value: "Bueno" if value == 1.0 else "Deficiente")
        property_area = st.selectbox("Property_Area", ["Urban", "Semiurban", "Rural"])

    return {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }


apply_styles()

st.markdown(
    """
    <div class="sim-hero">
        <h1>Simulador Bancario de Prestamos</h1>
        <p>Analiza una solicitud con todos los modelos entrenados y compara sus probabilidades de aprobacion.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

models, scaler, encoder, metadata, missing_files = load_artifacts()

if missing_files:
    st.warning(
        "Faltan artefactos entrenados en models/: "
        + ", ".join(missing_files)
    )

if not models or scaler is None or encoder is None or metadata is None:
    st.info(
        "Ejecuta primero el notebook entrenamiento.ipynb para generar los modelos, "
        "scaler.pkl, encoder.pkl y columns.pkl. Esta pagina no entrena modelos."
    )
    st.stop()

form_values = render_form()

analyze = st.button("Analizar Solicitud", type="primary", use_container_width=True)

if analyze:
    try:
        input_record = build_input_record(form_values, metadata)
        transformed_record = preprocess_record(input_record, scaler, encoder, metadata)

        predictions = []
        for model_name, model in models.items():
            start_time = perf_counter()
            probability = get_probability(model, transformed_record)
            elapsed_ms = (perf_counter() - start_time) * 1000

            predictions.append(
                {
                    "Modelo": model_name,
                    "Prediccion": get_prediction_label(probability),
                    "Probabilidad": probability,
                    "Tiempo de respuesta": elapsed_ms,
                }
            )

        results_df = pd.DataFrame(predictions).sort_values("Probabilidad", ascending=False).reset_index(drop=True)
        recommended_row = results_df.iloc[0]
        average_probability = float(results_df["Probabilidad"].mean())
        final_decision = get_prediction_label(average_probability)

        render_decision_card(
            final_decision=final_decision,
            recommended_model=recommended_row["Modelo"],
            probability=average_probability,
        )

        st.markdown("### Resultado por modelo")
        st.dataframe(
            results_df.assign(
                Probabilidad=results_df["Probabilidad"].map(lambda value: f"{value:.2%}"),
                **{
                    "Tiempo de respuesta": results_df["Tiempo de respuesta"].map(lambda value: f"{value:.2f} ms")
                },
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Comparacion de probabilidades")
        fig = px.bar(
            results_df,
            x="Modelo",
            y="Probabilidad",
            color="Prediccion",
            text=results_df["Probabilidad"].map(lambda value: f"{value:.1%}"),
            title="Probabilidad de aprobacion por modelo",
            color_discrete_map={"Aprobado": "#0f766e", "Rechazado": "#dc2626"},
        )
        fig.update_layout(
            height=430,
            yaxis_tickformat=".0%",
            yaxis_range=[0, 1],
            margin=dict(l=20, r=20, t=60, b=55),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#334155"),
        )
        fig.update_yaxes(gridcolor="#e2e8f0")
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Modelo recomendado")
        st.success(
            f"{recommended_row['Modelo']} presenta la probabilidad individual mas alta "
            f"({recommended_row['Probabilidad']:.2%}) para esta solicitud."
        )

    except Exception as exc:
        st.error(f"No fue posible analizar la solicitud: {exc}")
else:
    st.info("Completa el formulario y pulsa Analizar Solicitud para consultar todos los modelos entrenados.")
