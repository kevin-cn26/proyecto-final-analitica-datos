from pathlib import Path

import joblib
import streamlit as st


st.set_page_config(
    page_title="Recomendaciones | Prestamos",
    page_icon=":bulb:",
    layout="wide",
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"
COLUMNS_PATH = MODELS_DIR / "columns.pkl"


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

            .recommendation-hero {
                background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
                border: 1px solid #bfdbfe;
                border-left: 6px solid var(--bank-blue);
                border-radius: 8px;
                padding: 1.65rem 1.85rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 18px 42px rgba(15, 23, 42, .08);
            }

            .recommendation-hero h1 {
                color: var(--bank-navy);
                margin: 0 0 .35rem 0;
            }

            .recommendation-hero p {
                color: var(--bank-slate);
                margin: 0;
                font-size: 1.02rem;
            }

            .status-card {
                border: 1px solid #bfdbfe;
                border-radius: 8px;
                padding: 1.25rem 1.35rem;
                background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
                box-shadow: 0 12px 30px rgba(37, 99, 235, 0.12);
                margin-bottom: 1rem;
            }

            .status-card.rejected {
                border-color: #fecaca;
                background: #fef2f2;
                box-shadow: 0 10px 28px rgba(220, 38, 38, 0.08);
            }

            .status-label {
                color: #475569;
                font-size: .84rem;
                font-weight: 700;
                text-transform: uppercase;
                margin-bottom: .3rem;
            }

            .status-value {
                color: #0f172a;
                font-size: 1.7rem;
                font-weight: 800;
                line-height: 1.15;
            }

            .status-note {
                color: #475569;
                margin-top: .4rem;
            }

            .action-card {
                border: 1px solid var(--bank-border);
                border-radius: 8px;
                padding: 1.2rem 1.25rem;
                background: var(--bank-card);
                box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
                min-height: 210px;
                margin-bottom: 1rem;
            }

            .action-icon {
                font-size: 1.65rem;
                margin-bottom: .45rem;
            }

            .action-title {
                color: #0f172a;
                font-weight: 750;
                font-size: 1.05rem;
                margin-bottom: .35rem;
            }

            .action-text {
                color: #475569;
                font-size: .94rem;
                line-height: 1.45;
                margin-bottom: .65rem;
            }

            .bank-help {
                color: #334155;
                background: #f8fafc;
                border-left: 4px solid #2563eb;
                border-radius: 6px;
                padding: .7rem .8rem;
                font-size: .9rem;
                line-height: 1.4;
            }

            div[data-testid="stSelectbox"] > div,
            div[data-testid="stSlider"] {
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
def load_best_model_name() -> str:
    """Lee el mejor modelo registrado por el notebook, si el artefacto existe."""
    if not COLUMNS_PATH.exists():
        return "Mejor modelo entrenado"

    metadata = joblib.load(COLUMNS_PATH)
    return metadata.get("best_model_by_f1", "Mejor modelo entrenado")


def get_model_result() -> tuple[str, float | None, str]:
    """Obtiene el resultado desde session_state o desde un selector de escenario."""
    stored_prediction = st.session_state.get("best_model_prediction")
    stored_probability = st.session_state.get("best_model_probability")
    stored_model = st.session_state.get("recommended_model")

    if stored_prediction in {"Aprobado", "Rechazado"}:
        return stored_prediction, stored_probability, stored_model or load_best_model_name()

    model_name = load_best_model_name()

    col_left, col_right = st.columns([1, 1])
    with col_left:
        selected_result = st.selectbox(
            "Resultado del mejor modelo",
            ["Aprobado", "Rechazado"],
            help="Cuando el simulador guarde el resultado, esta pagina lo usara automaticamente.",
        )

    with col_right:
        selected_probability = st.slider(
            "Probabilidad estimada",
            min_value=0,
            max_value=100,
            value=72 if selected_result == "Aprobado" else 38,
            step=1,
        )

    return selected_result, selected_probability / 100, model_name


def render_status_card(result: str, probability: float | None, model_name: str) -> None:
    css_class = "status-card" if result == "Aprobado" else "status-card rejected"
    probability_text = "No disponible" if probability is None else f"{probability:.2%}"

    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="status-label">Resultado del mejor modelo</div>
            <div class="status-value">Prestamo {result}</div>
            <div class="status-note">
                Modelo de referencia: <strong>{model_name}</strong> |
                Probabilidad estimada: <strong>{probability_text}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_action_card(icon: str, title: str, action: str, bank_reason: str) -> None:
    st.markdown(
        f"""
        <div class="action-card">
            <div class="action-icon">{icon}</div>
            <div class="action-title">{title}</div>
            <div class="action-text">{action}</div>
            <div class="bank-help"><strong>Ayuda al banco:</strong> {bank_reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


APPROVED_RECOMMENDATIONS = [
    {
        "icon": "✅",
        "title": "Validar documentacion",
        "action": "Revisar identidad, ingresos declarados, historial laboral y documentos de respaldo antes de pasar a aprobacion operativa.",
        "bank_reason": "reduce riesgo documental y confirma que la informacion usada por el modelo coincide con evidencia verificable.",
    },
    {
        "icon": "📄",
        "title": "Preparar condiciones del credito",
        "action": "Definir monto final, plazo, tasa, calendario de pagos y condiciones contractuales para presentarlas al solicitante.",
        "bank_reason": "permite convertir una buena prediccion en una oferta clara, rentable y alineada con politicas internas.",
    },
    {
        "icon": "🏦",
        "title": "Realizar revision de riesgo final",
        "action": "Ejecutar validaciones internas de cumplimiento, endeudamiento y exposicion antes del desembolso.",
        "bank_reason": "mantiene control regulatorio y evita aprobar casos con alertas no capturadas por variables historicas.",
    },
    {
        "icon": "📊",
        "title": "Monitorear comportamiento temprano",
        "action": "Registrar el caso para seguimiento durante los primeros pagos y detectar atrasos de forma preventiva.",
        "bank_reason": "mejora la gestion de cartera y permite acciones tempranas antes de que el credito se deteriore.",
    },
]

REJECTED_RECOMMENDATIONS = [
    {
        "icon": "💳",
        "title": "Fortalecer historial crediticio",
        "action": "Recomendar pagos puntuales, reduccion de moras y uso responsable de productos financieros antes de una nueva solicitud.",
        "bank_reason": "disminuye la probabilidad de incumplimiento y mejora la calidad esperada de futuros solicitantes.",
    },
    {
        "icon": "📉",
        "title": "Reducir el monto solicitado",
        "action": "Evaluar un monto menor o un plazo mas amplio para que la cuota sea compatible con la capacidad de pago.",
        "bank_reason": "reduce exposicion crediticia y mejora la relacion entre ingreso, cuota y nivel de riesgo.",
    },
    {
        "icon": "👥",
        "title": "Agregar co-solicitante solvente",
        "action": "Considerar un co-solicitante con ingresos estables o mejor historial para reforzar el perfil financiero.",
        "bank_reason": "diversifica la fuente de repago y aumenta la capacidad de absorcion ante eventos adversos.",
    },
    {
        "icon": "🧾",
        "title": "Completar informacion faltante",
        "action": "Solicitar documentos de ingresos, empleo, activos o garantias que reduzcan incertidumbre sobre la solicitud.",
        "bank_reason": "mejora la calidad de datos y permite decisiones mas justas, auditables y consistentes.",
    },
]


apply_styles()

st.markdown(
    """
    <div class="recommendation-hero">
        <h1>Recomendaciones Crediticias</h1>
        <p>Acciones sugeridas segun el resultado del mejor modelo y su impacto para la gestion bancaria.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

result, probability, model_name = get_model_result()
render_status_card(result, probability, model_name)

if result == "Aprobado":
    st.markdown("### Recomendaciones para continuar el proceso")
    recommendations = APPROVED_RECOMMENDATIONS
else:
    st.markdown("### Recomendaciones para mejorar la solicitud")
    recommendations = REJECTED_RECOMMENDATIONS

row_1_col_1, row_1_col_2 = st.columns(2)
row_2_col_1, row_2_col_2 = st.columns(2)

with row_1_col_1:
    render_action_card(**recommendations[0])

with row_1_col_2:
    render_action_card(**recommendations[1])

with row_2_col_1:
    render_action_card(**recommendations[2])

with row_2_col_2:
    render_action_card(**recommendations[3])

st.markdown("### Enfoque de gestion")

if result == "Aprobado":
    st.success(
        "La prioridad es convertir la aprobacion predictiva en una decision operativa segura: "
        "validar datos, formalizar condiciones y mantener control de riesgo antes del desembolso."
    )
else:
    st.warning(
        "La prioridad es reducir incertidumbre y riesgo: mejorar capacidad de pago, documentacion, "
        "historial crediticio y estructura de la solicitud antes de volver a evaluarla."
    )
