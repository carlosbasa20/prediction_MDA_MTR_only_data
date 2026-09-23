import hmac
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Predicción de precios",
    layout="wide"
)

# ─────────────────────────────
# Contraseña
# ─────────────────────────────
if not st.session_state.get("autorizado"):

    st.title("Dashboard de modelos")

    clave = st.text_input("Contraseña", type="password")

    if st.button("Entrar"):
        if hmac.compare_digest(
            clave,
            st.secrets["APP_PASSWORD"]
        ):
            st.session_state["autorizado"] = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta")

    st.stop()


# ─────────────────────────────
# Cargar datos
# ─────────────────────────────
@st.cache_data
def cargar_datos(archivo):

    datos = pd.read_csv(
        archivo,
        encoding="utf-8-sig"
    )

    datos.columns = datos.columns.str.strip()

    # Permitir ambas versiones del nombre de la columna
    if "Tiempo de predicción" in datos.columns:
        datos = datos.rename(
            columns={
                "Tiempo de predicción": "tiempo de predicción"
            }
        )

    columnas = [
        "modelo",
        "mercado",
        "tiempo de predicción",
        "FechaHora_objetivo",
        "valor_real",
        "Q10",
        "Q50",
        "Q90",
        "dentro_del_rango"
    ]

    if not set(columnas).issubset(datos.columns):
        st.error(
            f"{archivo} debe contener estas columnas:"
        )
        st.write(columnas)
        st.write(
            "Columnas encontradas:",
            list(datos.columns)
        )
        st.stop()

    datos["FechaHora_objetivo"] = pd.to_datetime(
        datos["FechaHora_objetivo"],
        errors="coerce"
    )

    columnas_numericas = [
        "tiempo de predicción",
        "valor_real",
        "Q10",
        "Q50",
        "Q90"
    ]

    for columna in columnas_numericas:
        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce"
        )

    # Convertir dentro_del_rango de forma robusta
    if datos["dentro_del_rango"].dtype != bool:
        datos["dentro_del_rango"] = (
            datos["dentro_del_rango"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False
            })
        )

    datos = datos.dropna(
        subset=[
            "FechaHora_objetivo",
            "mercado",
            "tiempo de predicción",
            "valor_real",
            "Q10",
            "Q50",
            "Q90",
            "dentro_del_rango"
        ]
    )

    return datos


# ─────────────────────────────
# Archivo
# ─────────────────────────────
datos = cargar_datos(
    "HGB_predicciones_detalle.csv"
)


# ─────────────────────────────
# Título
# ─────────────────────────────
st.title(
    "Predicción de precios MDA y MTR"
)


# ─────────────────────────────
# Selectores
# ─────────────────────────────
col_selector1, col_selector2 = st.columns(2)

mercados = sorted(
    datos["mercado"].dropna().unique()
)

mercado = col_selector1.selectbox(
    "Mercado",
    mercados
)


datos_mercado = datos[
    datos["mercado"] == mercado
].copy()


tiempos = sorted(
    datos_mercado["tiempo de predicción"]
    .dropna()
    .unique()
)

tiempo = col_selector2.selectbox(
    "Tiempo de predicción",
    tiempos,
    format_func=lambda valor: f"{valor:g} horas"
)


# Datos del mercado y horizonte seleccionados
datos_horizonte = datos_mercado[
    datos_mercado["tiempo de predicción"] == tiempo
].copy()

datos_horizonte = datos_horizonte.sort_values(
    "FechaHora_objetivo"
)


if datos_horizonte.empty:
    st.error(
        "No hay datos para esta combinación."
    )
    st.stop()


# ─────────────────────────────
# Métricas generales
# ─────────────────────────────
total_horas = len(
    datos_horizonte
)

horas_dentro = int(
    datos_horizonte["dentro_del_rango"].sum()
)

porcentaje_dentro = (
    horas_dentro
    / total_horas
    * 100
)

mae = np.mean(
    np.abs(
        datos_horizonte["valor_real"]
        - datos_horizonte["Q50"]
    )
)

rmse = np.sqrt(
    np.mean(
        (
            datos_horizonte["valor_real"]
            - datos_horizonte["Q50"]
        ) ** 2
    )
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Horas dentro del rango",
    f"{horas_dentro:,} de {total_horas:,}"
)

col2.metric(
    "Cobertura Q10–Q90",
    f"{porcentaje_dentro:.1f}%"
)

col3.metric(
    "MAE",
    f"${mae:,.2f}/MWh"
)

col4.metric(
    "RMSE",
    f"${rmse:,.2f}/MWh"
)


# ─────────────────────────────
# Selector de fechas para la gráfica
# ─────────────────────────────
fecha_min = (
    datos_horizonte["FechaHora_objetivo"]
    .min()
    .date()
)

fecha_max = (
    datos_horizonte["FechaHora_objetivo"]
    .max()
    .date()
)


st.subheader(
    "Evolución temporal"
)

periodo = st.date_input(
    "Periodo mostrado en la gráfica",
    value=(fecha_min, fecha_max),
    min_value=fecha_min,
    max_value=fecha_max
)


if isinstance(periodo, tuple) and len(periodo) == 2:

    fecha_inicio = pd.Timestamp(
        periodo[0]
    )

    fecha_fin = (
        pd.Timestamp(periodo[1])
        + pd.Timedelta(days=1)
    )

else:
    fecha_inicio = pd.Timestamp(
        fecha_min
    )

    fecha_fin = (
        pd.Timestamp(fecha_max)
        + pd.Timedelta(days=1)
    )


# Quitar zona horaria para comparar si fuera necesario
fechas_comparacion = (
    datos_horizonte["FechaHora_objetivo"]
    .dt.tz_localize(None)
    if datos_horizonte["FechaHora_objetivo"].dt.tz is not None
    else datos_horizonte["FechaHora_objetivo"]
)


datos_grafica = datos_horizonte[
    (fechas_comparacion >= fecha_inicio)
    & (fechas_comparacion < fecha_fin)
].copy()


if datos_grafica.empty:
    st.warning(
        "No hay datos en el periodo seleccionado."
    )
    st.stop()


# ─────────────────────────────
# Gráfica
# ─────────────────────────────
fig = go.Figure()


# Límite superior Q90
fig.add_trace(
    go.Scatter(
        x=datos_grafica["FechaHora_objetivo"],
        y=datos_grafica["Q90"],
        mode="lines",
        line=dict(width=0),
        name="Q90",
        hovertemplate=(
            "Q90: $%{y:,.2f}/MWh"
            "<extra></extra>"
        )
    )
)


# Límite inferior Q10 + sombreado
fig.add_trace(
    go.Scatter(
        x=datos_grafica["FechaHora_objetivo"],
        y=datos_grafica["Q10"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(31, 119, 180, 0.20)",
        name="Rango Q10–Q90",
        hovertemplate=(
            "Q10: $%{y:,.2f}/MWh"
            "<extra></extra>"
        )
    )
)


# Predicción central Q50
fig.add_trace(
    go.Scatter(
        x=datos_grafica["FechaHora_objetivo"],
        y=datos_grafica["Q50"],
        mode="lines",
        name="Predicción Q50",
        line=dict(
            color="#ff7f0e",
            width=1.5
        ),
        hovertemplate=(
            "Q50: $%{y:,.2f}/MWh"
            "<extra></extra>"
        )
    )
)


# Precio real
fig.add_trace(
    go.Scatter(
        x=datos_grafica["FechaHora_objetivo"],
        y=datos_grafica["valor_real"],
        mode="lines",
        name="Precio real",
        line=dict(
            color="black",
            width=1.5
        ),
        hovertemplate=(
            "Real: $%{y:,.2f}/MWh"
            "<extra></extra>"
        )
    )
)


# Puntos que quedaron FUERA del intervalo
fuera = datos_grafica[
    ~datos_grafica["dentro_del_rango"]
]


fig.add_trace(
    go.Scatter(
        x=fuera["FechaHora_objetivo"],
        y=fuera["valor_real"],
        mode="markers",
        name="Fuera del rango",
        marker=dict(
            color="red",
            size=5
        ),
        hovertemplate=(
            "Fuera del rango"
            "<br>Real: $%{y:,.2f}/MWh"
            "<extra></extra>"
        )
    )
)


fig.update_layout(
    title=(
        f"{mercado} — "
        f"{tiempo:g} horas de anticipación"
    ),
    xaxis_title="Fecha",
    yaxis_title="Precio ($/MWh)",
    hovermode="x unified",
    legend_title="Serie"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ─────────────────────────────
# Cobertura del periodo mostrado
# ─────────────────────────────
horas_periodo = len(
    datos_grafica
)

horas_dentro_periodo = int(
    datos_grafica["dentro_del_rango"].sum()
)

cobertura_periodo = (
    horas_dentro_periodo
    / horas_periodo
    * 100
)


st.caption(
    f"En el periodo mostrado, "
    f"{horas_dentro_periodo:,} de {horas_periodo:,} horas "
    f"({cobertura_periodo:.1f}%) quedaron dentro de Q10–Q90."
)


# ─────────────────────────────
# Tabla opcional
# ─────────────────────────────
st.subheader(
    "Datos del periodo"
)

tabla = datos_grafica[
    [
        "FechaHora_objetivo",
        "valor_real",
        "Q10",
        "Q50",
        "Q90",
        "dentro_del_rango"
    ]
].copy()


tabla = tabla.rename(
    columns={
        "FechaHora_objetivo": "Fecha objetivo",
        "valor_real": "Precio real",
        "dentro_del_rango": "Dentro del rango"
    }
)


st.dataframe(
    tabla,
    use_container_width=True,
    hide_index=True
)
