# Sistema Inteligente para la Prediccion de Aprobacion de Prestamos

Proyecto profesional de analitica de datos desarrollado con Python y Streamlit.

## Objetivo

Construir una aplicacion interactiva para analizar solicitudes de prestamo y,
en una fase posterior, incorporar modelos de Machine Learning que permitan
estimar la aprobacion de una solicitud.

No incluye todavia:

- Carga de datos
- Limpieza o transformacion de datos
- Entrenamiento de modelos
- Predicciones
- Funcionalidad de simulacion real

## Estructura del proyecto

```text
Proyecto_Analitica_Datos/
├── app.py
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Modelos.py
│   ├── 3_Simulador.py
│   ├── 4_Recomendaciones.py
│   └── 5_Acerca.py
├── models/
├── dataset/
├── utils/
├── images/
├── notebooks/
├── requirements.txt
└── README.md
```

## Instalacion

```bash
pip install -r requirements.txt
```

## Ejecucion

```bash
streamlit run app.py
```

## Paginas

- **Dashboard:** resumen ejecutivo y espacio para visualizaciones.
- **Modelos:** base para comparar modelos predictivos en fases posteriores.
- **Simulador:** interfaz inicial para evaluar solicitudes de prestamo.
- **Recomendaciones:** espacio para sugerencias y criterios de decision.
- **Acerca:** descripcion general del proyecto.

## Proximas fases

1. Incorporar dataset de solicitudes de prestamo.
2. Realizar analisis exploratorio de datos.
3. Preparar variables para modelado.
4. Entrenar y comparar modelos de Machine Learning.
5. Integrar predicciones al simulador.
