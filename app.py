import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from math import factorial
from pathlib import Path

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import tensorflow as tf


# ==================================================
# 1. CONFIGURACIÓN
# ==================================================

st.set_page_config(
    page_title="Airbnb Bogotá",
    page_icon="🏠"
)

RAIZ = Path(__file__).resolve().parent

ARCHIVOS = [
    "preprocesador_final.pkl",
    "modelo_final_mlp.keras",
    "fondo_shap.npz",
    "metadata.json"
]

NOMBRES = [
    "Descripción del alojamiento",
    "Capacidad de huéspedes",
    "Tipo de alojamiento"
]

TRADUCCION = {
    "Entire home/apt": "Alojamiento completo",
    "Private room": "Habitación privada",
    "Shared room": "Habitación compartida",
    "Hotel room": "Habitación de hotel"
}


# ==================================================
# 2. CARGAR EL MODELO
# ==================================================

@st.cache_resource
def cargar():
    with (RAIZ / "preprocesador_final.pkl").open("rb") as f:
        preprocesador = cloudpickle.load(f)

    modelo = tf.keras.models.load_model(
        RAIZ / "modelo_final_mlp.keras",
        compile=False
    )

    fondo = np.load(
        RAIZ / "fondo_shap.npz"
    )["fondo"]

    metadata = json.loads(
        (RAIZ / "metadata.json").read_text(encoding="utf-8")
    )

    return preprocesador, modelo, fondo, metadata


# ==================================================
# 3. EXPLICACIÓN INDIVIDUAL
# ==================================================

def explicar_prediccion(modelo, x, fondo, bloques):
    """
    Calcula valores Shapley para los tres bloques:
    descripción, capacidad y tipo de alojamiento.

    Los bloques ausentes se reemplazan por el fondo.
    Los aportes están en escala de probabilidad.
    """
    x = np.asarray(x, dtype=np.float32).reshape(1, -1)
    fondo = np.asarray(fondo, dtype=np.float32)

    n = len(bloques)
    lotes = []

    for mascara in range(2 ** n):
        mezcla = fondo.copy()

        for j, columnas in enumerate(bloques):
            if mascara & (1 << j):
                mezcla[:, columnas] = x[:, columnas]

        lotes.append(mezcla)

    probabilidades = (
        modelo(
            np.concatenate(lotes),
            training=False
        )
        .numpy()
        .ravel()
    )

    valores = (
        probabilidades
        .reshape(2 ** n, len(fondo))
        .mean(axis=1)
    )

    aportes = np.zeros(n)

    for j in range(n):
        for mascara in range(2 ** n):
            if mascara & (1 << j):
                continue

            cantidad = mascara.bit_count()

            peso = (
                factorial(cantidad)
                * factorial(n - cantidad - 1)
                / factorial(n)
            )

            aportes[j] += peso * (
                valores[mascara | (1 << j)]
                - valores[mascara]
            )

    return float(valores[0]), aportes


# ==================================================
# 4. FORMULARIO
# ==================================================

st.title("¿Conseguirá su primera reseña en 30 días?")

st.write(
    "Ingresa las características de un alojamiento "
    "de Airbnb en Bogotá."
)

faltantes = [
    archivo
    for archivo in ARCHIVOS
    if not (RAIZ / archivo).exists()
]

if faltantes:
    st.error(
        "Faltan archivos del entrenamiento: "
        + ", ".join(faltantes)
    )
    st.info("Ejecuta primero entrenar_modelo.py.")
    st.stop()

try:
    preprocesador, modelo, fondo, metadata = cargar()
except Exception as error:
    st.error(
        "No se pudo cargar el modelo. Revisa los archivos "
        "y las versiones de las dependencias."
    )
    st.exception(error)
    st.stop()

with st.form("alojamiento"):
    descripcion = st.text_area(
        "Descripción del alojamiento",
        placeholder=(
            "Describe la ubicación, características "
            "y comodidades del alojamiento."
        ),
        height=150,
        max_chars=12000
    )

    capacidad = st.number_input(
        "Capacidad de huéspedes",
        min_value=1,
        max_value=100,
        value=2,
        step=1
    )

    tipo = st.selectbox(
        "Tipo de alojamiento",
        metadata["room_types"],
        format_func=lambda valor: TRADUCCION.get(
            valor,
            valor
        )
    )

    enviar = st.form_submit_button("Predecir")


# ==================================================
# 5. PREDICCIÓN
# ==================================================

if enviar:
    if not descripcion.strip():
        st.warning("Escribe la descripción del alojamiento.")
        st.stop()

    fila = pd.DataFrame([{
        "description": descripcion.strip(),
        "accommodates": capacidad,
        "room_type": tipo
    }])

    with st.spinner("Calculando predicción y explicación…"):
        x = np.asarray(
            preprocesador.transform(fila),
            dtype=np.float32
        )

        p = float(
            modelo(x, training=False)
            .numpy()
            .ravel()[0]
        )

        base, aportes = explicar_prediccion(
            modelo,
            x,
            fondo,
            metadata["bloques"]
        )

    st.metric(
        "Probabilidad estimada de primera reseña en 30 días",
        f"{p:.1%}"
    )

    st.metric(
        "Probabilidad estimada de no obtenerla en esa ventana",
        f"{1 - p:.1%}"
    )

    if p >= 0.5:
        st.success(
            "Clase 1: el modelo estima que obtendrá "
            "su primera reseña dentro de la ventana."
        )
    else:
        st.warning(
            "Clase 0: el modelo estima que no obtendrá "
            "su primera reseña dentro de la ventana. "
            "Podría priorizarse para apoyo."
        )

    st.caption(
        "Umbral de clasificación: 0.50, "
        "igual al utilizado para evaluar el modelo."
    )


    # ==============================================
    # 6. GRÁFICA DE EXPLICACIÓN
    # ==============================================

    st.subheader("¿Qué factores influyen en esta predicción?")

    explicacion = shap.Explanation(
        values=aportes,
        base_values=base,
        feature_names=NOMBRES
    )

    shap.plots.waterfall(
        explicacion,
        max_display=3,
        show=False
    )

    figura = plt.gcf()
    figura.set_size_inches(9, 4.5)

    st.pyplot(figura)
    plt.close(figura)

    st.caption(
        "Rojo: aumenta la probabilidad de obtener la reseña. "
        "Azul: la reduce. Los aportes están en probabilidad; "
        "0.10 equivale a 10 puntos porcentuales."
    )

    tabla = pd.DataFrame({
        "Factor": NOMBRES,
        "Aporte (puntos porcentuales)": np.round(
            aportes * 100,
            2
        ),
        "Efecto": [
            "Aumenta la probabilidad" if valor > 0
            else "Reduce la probabilidad" if valor < 0
            else "Sin aporte"
            for valor in aportes
        ]
    })

    tabla = (
        tabla.iloc[np.argsort(-np.abs(aportes))]
        .reset_index(drop=True)
    )

    st.dataframe(tabla, hide_index=True)

    with st.expander("Cómo interpretar la explicación"):
        st.write(
            "La referencia es el promedio de predicciones "
            "de una muestra fija de hasta 32 alojamientos "
            "del conjunto de desarrollo."
        )

        st.write(
            "Se evalúan las ocho combinaciones de los tres "
            "factores. La descripción agrupa sus componentes "
            "SVD y el tipo agrupa sus columnas OneHot. "
            "La explicación corresponde al caso ingresado "
            "y no demuestra causas."
        )

        st.write(
            f"Probabilidad de referencia: {base:.1%}"
        )

        st.write(
            "Referencia + aportes: "
            f"{base + aportes.sum():.1%}"
        )

        if not np.isclose(
            base + aportes.sum(),
            p,
            atol=1e-5
        ):
            st.warning(
                "Existe una diferencia numérica "
                "en la reconstrucción."
            )


# ==================================================
# 7. INFORMACIÓN DEL MODELO
# ==================================================

with st.expander("Modelo y limitaciones"):
    st.write(
        "Modelo MLP con capas de 64 y 32 neuronas, "
        "Dropout de 30 % y salida sigmoid. Utiliza "
        "descripción, capacidad y tipo de alojamiento."
    )

    st.write(
        f"F1 macro del test de este entrenamiento: "
        f"{metadata['f1_macro_test']:.4f}"
    )

    st.write(
        "El inicio del alojamiento se aproximó a partir "
        "de la antigüedad del anfitrión, porque los datos "
        "no contienen la fecha exacta de publicación. "
        "Se analizaron anfitriones con un alojamiento."
    )

    st.write(
        "La salida es una estimación sin calibración "
        "adicional y el entrenamiento utiliza pesos de clase. "
        "No garantiza que el alojamiento consiga una reseña."
    )
