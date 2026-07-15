from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Dashboard | Prestamos",
    page_icon=":bar_chart:",
    layout="wide",
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"
TARGET_COLUMN = "Loan_Status"


def apply_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bank-navy: #0f172a;
                --bank-blue: #1d4ed8;
                --bank-blue-soft: #dbeafe;
                --bank-slate: #475569;
                --bank-muted: #64748b;
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

            section[data-testid="stSidebar"] a {
                border-radius: 8px;
                margin: .08rem 0;
            }

            .dashboard-hero {
                background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
                border: 1px solid #bfdbfe;
                border-left: 6px solid var(--bank-blue);
                border-radius: 8px;
                padding: 1.65rem 1.85rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 18px 42px rgba(15, 23, 42, .08);
            }

            .dashboard-hero h1 {
                color: var(--bank-navy);
                margin: 0 0 .35rem 0;
            }

            .dashboard-hero p {
                color: var(--bank-slate);
                margin: 0;
                font-size: 1.02rem;
            }

            .kpi-card {
                border: 1px solid var(--bank-border);
                border-radius: 8px;
                padding: 1.1rem 1.15rem;
                background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
                min-height: 122px;
                position: relative;
                overflow: hidden;
            }

            .kpi-card:before {
                content: "";
                position: absolute;
                inset: 0 auto 0 0;
                width: 4px;
                background: var(--bank-blue);
            }

            .kpi-label {
                color: var(--bank-muted);
                font-size: .86rem;
                margin-bottom: .35rem;
            }

            .kpi-value {
                color: var(--bank-navy);
                font-size: 2rem;
                font-weight: 750;
                line-height: 1.1;
            }

            .kpi-note {
                color: #64748b;
                font-size: .8rem;
                margin-top: .35rem;
            }

            div[data-testid="stPlotlyChart"] {
                border: 1px solid var(--bank-border);
                border-radius: 8px;
                padding: .5rem;
                background: var(--bank-card);
                box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
            }

            div[data-testid="stSelectbox"] > div {
                background: #ffffff;
                border-radius: 8px;
            }

            div[data-testid="stExpander"] {
                border: 1px solid var(--bank-border);
                border-radius: 8px;
                background: #ffffff;
                box-shadow: 0 8px 22px rgba(15, 23, 42, .04);
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


def normalize_target(value):
    """Normaliza etiquetas comunes del target a Aprobado/Rechazado."""
    if pd.isna(value):
        return "Sin dato"

    normalized = str(value).strip().lower()
    approved_values = {"y", "yes", "approved", "aprobado", "1", "true"}
    rejected_values = {"n", "no", "rejected", "rechazado", "0", "false"}

    if normalized in approved_values:
        return "Aprobado"
    if normalized in rejected_values:
        return "Rechazado"
    return str(value)


def format_number(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def render_kpi(label: str, value: int, note: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{format_number(value)}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    existing = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in existing:
            return existing[candidate.lower()]
    return None


def plot_categorical_distribution(df: pd.DataFrame, column: str, title: str):
    counts = (
        df[column]
        .fillna("Sin dato")
        .astype(str)
        .value_counts()
        .reset_index()
    )
    counts.columns = [column, "Cantidad"]

    fig = px.bar(
        counts,
        x=column,
        y="Cantidad",
        text="Cantidad",
        title=title,
        color=column,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        showlegend=False,
        height=390,
        margin=dict(l=20, r=20, t=60, b=40),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#334155"),
    )
    fig.update_yaxes(gridcolor="#e2e8f0")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def plot_numeric_histogram(df: pd.DataFrame, column: str, title: str):
    fig = px.histogram(
        df,
        x=column,
        nbins=32,
        title=title,
        marginal="box",
        color_discrete_sequence=["#1d4ed8"],
    )
    fig.update_layout(
        height=390,
        margin=dict(l=20, r=20, t=60, b=40),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#334155"),
    )
    fig.update_yaxes(gridcolor="#e2e8f0")
    return fig


def plot_boxplot(df: pd.DataFrame, column: str, title: str, target_column: str | None):
    if target_column:
        fig = px.box(
            df,
            x=target_column,
            y=column,
            color=target_column,
            title=title,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
    else:
        fig = px.box(
            df,
            y=column,
            title=title,
            color_discrete_sequence=["#2563eb"],
        )

    fig.update_layout(
        showlegend=False,
        height=390,
        margin=dict(l=20, r=20, t=60, b=40),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#334155"),
    )
    fig.update_yaxes(gridcolor="#e2e8f0")
    return fig


apply_styles()

st.markdown(
    """
    <div class="dashboard-hero">
        <h1>Dashboard de Prestamos</h1>
        <p>Vista ejecutiva e interactiva del Loan Prediction Dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = load_dataset()

if df.empty:
    st.warning(
        "No se encontro ningun archivo CSV en la carpeta dataset/. "
        "Agrega el Loan Prediction Dataset para activar los KPIs y graficos."
    )
    st.info(
        "Nombres soportados: loan_prediction.csv, LoanPrediction.csv, train.csv, "
        "train_u6lujuX_CVtuZ9i.csv, loan_data.csv o cualquier otro .csv dentro de dataset/."
    )
    st.stop()

df.columns = df.columns.str.strip()

target_column = find_column(df, [TARGET_COLUMN, "Target", "Status", "LoanStatus"])
display_df = df.copy()

if target_column:
    display_df["Target_Normalizado"] = display_df[target_column].apply(normalize_target)
    approved_count = int((display_df["Target_Normalizado"] == "Aprobado").sum())
    rejected_count = int((display_df["Target_Normalizado"] == "Rechazado").sum())
else:
    approved_count = 0
    rejected_count = 0

total_records = int(df.shape[0])
total_variables = int(df.shape[1])

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)

with kpi_1:
    render_kpi("Cantidad de registros", total_records, "Filas disponibles")

with kpi_2:
    render_kpi("Cantidad de variables", total_variables, "Columnas del dataset")

with kpi_3:
    render_kpi("Prestamos aprobados", approved_count, "Target positivo")

with kpi_4:
    render_kpi("Prestamos rechazados", rejected_count, "Target negativo")

st.markdown("### Distribucion general")

left_col, right_col = st.columns([1, 1])

with left_col:
    if target_column:
        target_counts = (
            display_df["Target_Normalizado"]
            .value_counts()
            .rename_axis("Estado")
            .reset_index(name="Cantidad")
        )
        fig_target = px.pie(
            target_counts,
            names="Estado",
            values="Cantidad",
            title="Distribucion del Target",
            hole=0.45,
            color="Estado",
            color_discrete_map={
                "Aprobado": "#14b8a6",
                "Rechazado": "#ef4444",
                "Sin dato": "#94a3b8",
            },
        )
        fig_target.update_layout(
            height=410,
            margin=dict(l=20, r=20, t=60, b=30),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#334155"),
        )
        st.plotly_chart(fig_target, use_container_width=True)
    else:
        st.info("No se encontro una columna target para graficar su distribucion.")

with right_col:
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_columns:
        selected_histogram = st.selectbox(
            "Variable numerica para histograma",
            numeric_columns,
            index=0,
        )
        st.plotly_chart(
            plot_numeric_histogram(
                df,
                selected_histogram,
                f"Histograma de {selected_histogram}",
            ),
            use_container_width=True,
        )
    else:
        st.info("No se encontraron variables numericas para histogramas.")

st.markdown("### Analisis numerico")

num_left, num_right = st.columns([1, 1])

with num_left:
    income_column = find_column(df, ["ApplicantIncome", "Income", "Ingreso", "Ingresos"])
    if income_column:
        st.plotly_chart(
            plot_numeric_histogram(
                df,
                income_column,
                "Distribucion de ingresos",
            ),
            use_container_width=True,
        )
    else:
        st.info("No se encontro una columna de ingresos.")

with num_right:
    loan_amount_column = find_column(df, ["LoanAmount", "Monto", "Loan_Amount", "MontoSolicitado"])
    if loan_amount_column:
        st.plotly_chart(
            plot_numeric_histogram(
                df,
                loan_amount_column,
                "Distribucion del monto solicitado",
            ),
            use_container_width=True,
        )
    else:
        st.info("No se encontro una columna de monto solicitado.")

box_left, box_right = st.columns([1, 1])

with box_left:
    if numeric_columns:
        selected_boxplot = st.selectbox(
            "Variable numerica para boxplot",
            numeric_columns,
            index=min(1, len(numeric_columns) - 1),
        )
        st.plotly_chart(
            plot_boxplot(
                display_df,
                selected_boxplot,
                f"Boxplot de {selected_boxplot}",
                "Target_Normalizado" if target_column else None,
            ),
            use_container_width=True,
        )

with box_right:
    if len(numeric_columns) >= 2:
        corr = df[numeric_columns].corr(numeric_only=True)
        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Corr."),
            )
        )
        fig_heatmap.update_layout(
            title="Heatmap de correlaciones",
            height=390,
            margin=dict(l=20, r=20, t=60, b=40),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#334155"),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Se requieren al menos dos variables numericas para el heatmap.")

st.markdown("### Analisis categorico")

gender_column = find_column(df, ["Gender", "Genero", "Sexo"])
education_column = find_column(df, ["Education", "Educacion"])
married_column = find_column(df, ["Married", "Estado_Civil", "EstadoCivil"])
credit_history_column = find_column(df, ["Credit_History", "CreditHistory", "Historial_Crediticio"])

cat_row_1_left, cat_row_1_right = st.columns([1, 1])

with cat_row_1_left:
    if gender_column:
        st.plotly_chart(
            plot_categorical_distribution(df, gender_column, "Distribucion de genero"),
            use_container_width=True,
        )
    else:
        st.info("No se encontro una columna de genero.")

with cat_row_1_right:
    if education_column:
        st.plotly_chart(
            plot_categorical_distribution(df, education_column, "Distribucion de educacion"),
            use_container_width=True,
        )
    else:
        st.info("No se encontro una columna de educacion.")

cat_row_2_left, cat_row_2_right = st.columns([1, 1])

with cat_row_2_left:
    if married_column:
        st.plotly_chart(
            plot_categorical_distribution(df, married_column, "Distribucion por estado civil"),
            use_container_width=True,
        )
    else:
        st.info("No se encontro una columna de estado civil.")

with cat_row_2_right:
    if credit_history_column:
        st.plotly_chart(
            plot_categorical_distribution(
                df,
                credit_history_column,
                "Distribucion de historial crediticio",
            ),
            use_container_width=True,
        )
    else:
        st.info("No se encontro una columna de historial crediticio.")

with st.expander("Vista previa del dataset", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)