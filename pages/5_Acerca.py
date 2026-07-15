import streamlit as st


st.set_page_config(
    page_title="Acerca | Prestamos",
    page_icon=":information_source:",
    layout="wide",
)


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

            .about-hero {
                background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
                border: 1px solid #bfdbfe;
                border-left: 6px solid var(--bank-blue);
                border-radius: 8px;
                padding: 1.65rem 1.85rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 18px 42px rgba(15, 23, 42, .08);
            }

            .about-hero h1 {
                color: var(--bank-navy);
                margin: 0 0 .4rem 0;
            }

            .about-hero p {
                color: var(--bank-slate);
                margin: 0;
                font-size: 1.03rem;
                line-height: 1.5;
            }

            .info-card {
                border: 1px solid var(--bank-border);
                border-radius: 8px;
                padding: 1.2rem 1.3rem;
                background: var(--bank-card);
                box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
                min-height: 178px;
                margin-bottom: 1rem;
            }

            .info-icon {
                font-size: 1.55rem;
                margin-bottom: .45rem;
            }

            .info-title {
                color: #0f172a;
                font-size: 1.06rem;
                font-weight: 780;
                margin-bottom: .4rem;
            }

            .info-text {
                color: #475569;
                line-height: 1.48;
                font-size: .95rem;
            }

            .tag-grid {
                display: flex;
                flex-wrap: wrap;
                gap: .55rem;
                margin-top: .25rem;
            }

            .tag {
                border: 1px solid #bfdbfe;
                border-radius: 999px;
                background: #eff6ff;
                color: #1e3a8a;
                padding: .42rem .72rem;
                font-size: .9rem;
                font-weight: 650;
            }

            .institution-card {
                border: 1px solid #bfdbfe;
                border-radius: 8px;
                background: #ffffff;
                padding: 1.2rem 1.35rem;
                box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
            }

            .institution-label {
                color: #1d4ed8;
                font-size: .82rem;
                font-weight: 800;
                text-transform: uppercase;
                margin-bottom: .25rem;
            }

            .institution-value {
                color: #0f172a;
                font-size: 1.03rem;
                font-weight: 720;
                margin-bottom: .7rem;
            }

            h1, h2, h3 {
                color: var(--bank-navy);
                letter-spacing: 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_card(icon: str, title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-icon">{icon}</div>
            <div class="info-title">{title}</div>
            <div class="info-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tags(items: list[str]) -> None:
    tags = "".join(f'<span class="tag">{item}</span>' for item in items)
    st.markdown(f'<div class="tag-grid">{tags}</div>', unsafe_allow_html=True)


apply_styles()

st.markdown(
    """
    <div class="about-hero">
        <h1>Sistema Inteligente para la Prediccion de Aprobacion de Prestamos</h1>
        <p>
            Aplicacion profesional de analitica de datos y Machine Learning
            orientada a apoyar la evaluacion de solicitudes crediticias mediante
            visualizacion, comparacion de modelos, simulacion y recomendaciones.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_description, col_objective = st.columns(2)

with col_description:
    render_card(
        "📌",
        "Descripcion del proyecto",
        "El sistema integra un flujo completo para analizar perfiles de solicitantes, "
        "evaluar modelos predictivos y presentar resultados de forma clara para la "
        "toma de decisiones en un contexto bancario.",
    )

with col_objective:
    render_card(
        "🎯",
        "Objetivo",
        "Predecir la aprobacion de prestamos a partir de variables financieras y "
        "personales, facilitando una evaluacion consistente, interpretable y util "
        "para usuarios de negocio.",
    )

st.markdown("### Dataset utilizado")
render_card(
    "🗂️",
    "Loan Prediction Dataset",
    "Se utiliza el Loan Prediction Dataset, que contiene informacion como genero, "
    "estado civil, dependientes, educacion, empleo independiente, ingresos, monto "
    "solicitado, plazo, historial crediticio, zona de propiedad y estado final del prestamo.",
)

st.markdown("### Algoritmos implementados")
render_tags(
    [
        "Logistic Regression",
        "Random Forest",
        "Support Vector Machine",
        "KNN",
        "Gaussian Naive Bayes",
    ]
)

st.markdown("### Tecnologias")
render_tags(
    [
        "Python",
        "Streamlit",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Plotly",
        "Joblib",
        "Jupyter Notebook",
    ]
)

st.markdown("### Informacion academica")

inst_col_1, inst_col_2, inst_col_3 = st.columns(3)

with inst_col_1:
    st.markdown(
        """
        <div class="institution-card">
            <div class="institution-label">Autor</div>
            <div class="institution-value">Kevin Custodio Niquen</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with inst_col_2:
    st.markdown(
        """
        <div class="institution-card">
            <div class="institution-label">Curso</div>
            <div class="institution-value">Analitica de Datos</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with inst_col_3:
    st.markdown(
        """
        <div class="institution-card">
            <div class="institution-label">Universidad</div>
            <div class="institution-value">Universidad Señor de Sipan</div>
        </div>
        """,
        unsafe_allow_html=True,
    )    
    