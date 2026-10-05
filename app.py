from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import numpy as np
import shap
import matplotlib.pyplot as plt


# --------------------------------------------------
# CARGAR EL MODELO
# --------------------------------------------------

st.set_page_config(
    page_title="Predicción de baja de clientes",
    page_icon="📊"
)

RUTA_MODELO = Path(__file__).resolve().parent / "modelo_fuga.joblib"


@st.cache_resource
def cargar():
    return joblib.load(RUTA_MODELO)


if not RUTA_MODELO.exists():
    st.error(
        "No se encontró modelo_fuga.joblib. "
        "Ejecuta primero el entrenamiento y coloca "
        "el archivo generado en la misma carpeta que app.py."
    )
    st.stop()

modelo = cargar()


# --------------------------------------------------
# FORMULARIO
# --------------------------------------------------

st.title("¿Se va este cliente?")

with st.form("cliente"):
    meses = st.slider(
        "Meses como cliente",
        min_value=0,
        max_value=72,
        value=12
    )

    cargo = st.number_input(
        "Cargo mensual ($)",
        min_value=18.0,
        max_value=119.0,
        value=70.0,
        step=5.0
    )

    contrato = st.selectbox(
        "Contrato",
        ["Month-to-Month", "One year", "Two year"]
    )

    internet = st.selectbox(
        "Internet",
        ["Fiber optic", "DSL", "No"]
    )

    pago = st.selectbox(
        "Método de pago",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    soporte = st.selectbox(
        "Soporte técnico",
        ["No", "Yes", "No internet service"]
    )

    seguridad = st.selectbox(
        "Seguridad en línea",
        ["No", "Yes", "No internet service"]
    )

    enviar = st.form_submit_button("Predecir")


# --------------------------------------------------
# PREDICCIÓN Y RESULTADO
# --------------------------------------------------

if enviar:
    # Evita combinaciones incompatibles con el servicio.
    if internet == "No":
        soporte = "No internet service"
        seguridad = "No internet service"

    elif (
        soporte == "No internet service"
        or seguridad == "No internet service"
    ):
        st.error(
            "Si el cliente tiene internet, selecciona "
            "'Yes' o 'No' para soporte y seguridad."
        )
        st.stop()

    fila = pd.DataFrame([{
        "tenure": meses,
        "MonthlyCharges": cargo,
        "Contract": contrato,
        "InternetService": internet,
        "PaymentMethod": pago,
        "TechSupport": soporte,
        "OnlineSecurity": seguridad
    }])

    p = float(modelo.predict_proba(fila)[0, 1])

    st.metric(
        "Probabilidad de que se dé de baja",
        f"{p:.1%}"
    )

    if p >= 0.5:
        st.error(
            "Riesgo ALTO: conviene contactarlo "
            "lo más pronto posible."
        )
    elif p >= 0.25:
        st.warning("Riesgo medio: vigilar.")
    else:
        st.success("Riesgo bajo.")

    st.caption(
        "Los límites de 25 % y 50 % son criterios "
        "ilustrativos para este ejercicio."
    )

        # --------------------------------------------------
    # EXPLICACIÓN SHAP DEL CLIENTE
    # --------------------------------------------------

    st.subheader("¿Qué factores influyen en esta predicción?")

    pre = modelo.named_steps["pre"]
    clf = modelo.named_steps["m"]

    # Transformar los datos del cliente.
    x = pre.transform(fila)

    # Para esta regresión logística, el promedio guardado
    # permite reproducir los aportes del fondo de entrenamiento.
    fondo = np.asarray(modelo.fondo_).reshape(1, -1)

    mascara = shap.maskers.Independent(
        fondo,
        max_samples=1
    )

    explicador = shap.LinearExplainer(clf, mascara)
    explicacion = explicador(x)

    valores = explicacion.values[0]
    base = float(np.asarray(explicacion.base_values).ravel()[0])

    # Agrupar las columnas OneHot por variable original.
    # Así mostramos "Contrato" en lugar de varias categorías.
    aportes = [
        float(valores[0]),
        float(valores[1])
    ]

    encoder = pre.named_transformers_["c"]
    inicio = pre.output_indices_["c"].start

    for categorias in encoder.categories_:
        fin = inicio + len(categorias)
        aportes.append(float(valores[inicio:fin].sum()))
        inicio = fin

    nombres = [
        "Meses como cliente",
        "Cargo mensual",
        "Contrato",
        "Internet",
        "Método de pago",
        "Soporte técnico",
        "Seguridad en línea"
    ]

    valores_cliente = [
        meses,
        cargo,
        contrato,
        internet,
        pago,
        soporte,
        seguridad
    ]

    etiquetas = [
        f"{nombre}: {valor}"
        for nombre, valor in zip(nombres, valores_cliente)
    ]

    explicacion_agrupada = shap.Explanation(
        values=np.asarray(aportes),
        base_values=base,
        feature_names=etiquetas
    )

    # Gráfica SHAP.
    shap.plots.waterfall(
        explicacion_agrupada,
        max_display=7,
        show=False
    )

    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Rojo: aumenta la estimación de baja respecto al "
        "cliente de referencia. Azul: la reduce. "
        "Los aportes están en log-odds, no en porcentajes."
    )

    # Tabla con los factores de mayor influencia.
    tabla = pd.DataFrame({
        "Factor": nombres,
        "Valor del cliente": [str(v) for v in valores_cliente],
        "Aporte SHAP": aportes
    })

    tabla["Efecto"] = [
        "Aumenta la estimación" if v > 0
        else "Reduce la estimación" if v < 0
        else "Sin aporte"
        for v in aportes
    ]

    orden = np.argsort(-np.abs(np.asarray(aportes)))
    tabla = tabla.iloc[orden].reset_index(drop=True)

    st.dataframe(
        tabla,
        hide_index=True
    )