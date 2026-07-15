import streamlit as st


st.set_page_config(
    page_title="Sistema Inteligente de Prestamos",
    page_icon=":briefcase:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_base_styles() -> None:
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
                padding-top: 1.4rem;
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

            section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
                padding-top: .6rem;
            }

            section[data-testid="stSidebar"] a {
                border-radius: 8px;
                padding: .35rem .6rem;
                margin: .1rem 0;
            }

            section[data-testid="stSidebar"] a:hover {
                background: rgba(219, 234, 254, .12);
            }

            .hero {
                border: 1px solid #c7d2fe;
                border-left: 6px solid var(--bank-blue);
                background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
                padding: 1.9rem 2rem;
                border-radius: 8px;
                margin-bottom: 1.5rem;
                box-shadow: 0 18px 42px rgba(15, 23, 42, .08);
            }

            .hero h1 {
                color: #0f172a;
                margin-bottom: .35rem;
            }

            .hero p {
                color: var(--bank-slate);
                font-size: 1.05rem;
                margin-bottom: 0;
            }

            .section-box {
                border: 1px solid var(--bank-border);
                border-radius: 8px;
                padding: 1.35rem 1.45rem;
                background: var(--bank-card);
                min-height: 150px;
                box-shadow: 0 12px 30px rgba(15, 23, 42, .06);
            }

            .muted {
                color: #64748b;
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


apply_base_styles()

st.sidebar.title("Navegacion")
st.sidebar.caption("Sistema inteligente para evaluacion crediticia")
st.sidebar.info(
    "Usa el menu lateral para recorrer el dashboard, modelos, simulador, "
    "recomendaciones y documentacion del proyecto."
)

st.markdown(
    """
    <div class="hero">
        <h1>Sistema Inteligente para la Prediccion de Aprobacion de Prestamos</h1>
        <p>
            Proyecto de analitica de datos orientado a estructurar,
            explorar y presentar una solucion futura de Machine Learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("Vista general")
    st.markdown(
        """
        <div class="section-box">
            <p class="muted">
                Esta aplicacion Streamlit esta preparada como base del proyecto.
                En esta primera version solo se define la arquitectura, la
                navegacion y el estilo visual inicial.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_right:
    st.subheader("Estado del proyecto")
    st.markdown(
        """
        <div class="section-box">
            <p><strong>Fase actual:</strong> Estructura Final.</p>
            <p class="muted">
                Integra la funcionalidad de datos, entrenamiento y prediccion.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
