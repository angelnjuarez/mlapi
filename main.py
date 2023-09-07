from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import io
import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import joblib
import base64


app = FastAPI()

# Configurar el middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Esto permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Esto permite todos los encabezados
)


# Cargar los modelos
modelo = joblib.load("./Modelos/Regresion_pat_def.pkl")
probabilidad = joblib.load("./Modelos/ProbAltaDefEdad.pkl")

# Diccionario de patologías
patologias_dict: [int, str] = {
    1: "aborto",
    2: "af perinatales",
    3: "asma",
    4: "bronquitis/ bronquiolitis aguda",
    5: "causas obtétricas indirec",
    6: "chagas",
    7: "colelitiasis/colecistitis",
    8: "compl at medica/quirúrgica",
    9: "compl embarazo",
    10: "compl parto",
    11: "compl puerperio",
    12: "covid-19",
    13: "def nutricionales",
    14: "dengue",
    15: "diabetes",
    16: "ecv",
    17: "enf apéndice",
    18: "enf crónica vri",
    19: "enf hipertensivas",
    20: "enf hígado",
    21: "enf infecciosas intestinales",
    22: "enf oido/ ap mastoides",
    23: "enf ojos",
    24: "enf piel/ tcs",
    25: "enf páncreas",
    26: "enf sangre/ org hematopoyético",
    27: "enf sist osteomuscular",
    28: "enf sist urinario",
    29: "enteritis/ colitis no infecciosa",
    30: "envenenamiento/ tóxico",
    31: "epilepsia",
    32: "epoc",
    33: "fac que influyen en la salud/ contacto con serv salud",
    34: "hallazgos clínicos/ laboratorio anormales",
    35: "hepatitis virales",
    36: "hernias",
    37: "hiperplasia próstata",
    38: "iam",
    39: "influenza/ neumonía",
    40: "ira sup",
    41: "its",
    42: "leucemia",
    43: "malf congénitas",
    44: "meningitis bacteriana",
    45: "meningitis viral",
    46: "obesidad",
    47: "otras afec obstétricas",
    48: "otras causas externas",
    49: "otras enf cardíacas",
    50: "otras enf de vrs",
    51: "otras enf del peritoneo",
    52: "otras enf endócrinas/ metabólicas",
    53: "otras enf infecciosas/ parasitarias",
    54: "otras enf sist circulatorio",
    55: "otras enf sist digestivo",
    56: "otras enf sist genitourinario",
    57: "otras enf sist nervioso",
    58: "otras enf sist respiratorio",
    59: "otras ira inferiores",
    60: "otros trastornos mentales",
    61: "otros tumores malignos",
    62: "parotiditis infecciosa",
    63: "parto",
    64: "quemadura/ congelamiento",
    65: "salpingitis/ ooforitis",
    66: "sarampión",
    67: "sec causas ext",
    68: "septicemias",
    69: "t cabeza/ cuello",
    70: "t mmss y mmii",
    71: "t múltiples",
    72: "t tórax/ abdomen/ lumbopelvis",
    73: "tos ferina",
    74: "trast por alcohol",
    75: "trast tiroides",
    76: "tuberculosis",
    77: "tumor genitales fem",
    78: "tumor genitales masc",
    79: "tumor in situ/ benigno",
    80: "tumor mama",
    81: "tumor org digestivos",
    82: "tumor org intratoráxicos",
    83: "tumor org urinario",
    84: "tumor próstata",
    85: "tumor útero",
    86: "ulcera gástrica/ duodenal",
    87: "varicela",
    88: "vih",
}


@app.get("/listapatologias/")
def get_patologias():
    patologias_json = json.dumps(patologias_dict)

    return patologias_json


@app.get("/predecirdefunciones/")
def pred_egreso(codPatologias: List[int]):
    # Crear un dataframe con los valores recibidos
    features = pd.DataFrame({"causa_egreso": codPatologias})

    # Realizar la predicción
    prediccion = modelo.predict(features)

    # Creamos un dataframe y reemplazamos los códigos por los nombres de las patologías
    data = {"Enfermedad": codPatologias, "Predicción": prediccion}
    df = pd.DataFrame(data)
    df["Enfermedad"] = df["Enfermedad"].replace(patologias_dict)

    # Calcular total de defunciones por enfermedad y total de casos
    def_counts = df.groupby("Enfermedad")["Predicción"].sum().reset_index()
    enf_counts = df["Enfermedad"].value_counts().reset_index()
    enf_counts.columns = ["Enfermedad", "Total Casos"]
    result_df = pd.merge(def_counts, enf_counts, on="Enfermedad", how="left")

    return result_df


@app.post("/predecir/")
def graf_egreso(codPatologias: List[int]):
    # Obtener el dataframe con la predicción
    result_df = pred_egreso(codPatologias)
    df = pd.DataFrame(result_df, columns=["Enfermedad", "Predicción", "Total Casos"])

    # Crear un gráfico de tabla utilizando seaborn y matplotlib
    plt.figure(figsize=(6, 3), dpi=100)
    sns.set(font_scale=0.8)
    sns.heatmap(
        df.pivot_table(index="Enfermedad", aggfunc="sum"),
        annot=True,
        fmt="g",
        cmap="Blues",
    )

    # Convertir la figura a una cadena base64
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', bbox_inches="tight")
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode()

    return img_base64


@app.get("/probabilidadaltadefuncion/")
def calc_probabilidad(edades: List[int]):
    # Crear un dataframe con los valores recibidos
    features = pd.DataFrame({"grupo_edad": edades})

    # Realizar la predicción
    prediccion = probabilidad.predict_proba(features)

    probabilidad_alta = []
    probabilidad_defuncion = []

    for i in range(len(edades)):
        probabilidad_alta.append(prediccion[i][0] * 100)
        probabilidad_defuncion.append(prediccion[i][1] * 100)

    # Crear una tabla con valores input y predicciones
    data = {
        "Edad": edades,
        "Probabilidad Alta": probabilidad_alta,
        "Probabilidad Defunción": probabilidad_defuncion,
    }
    df = pd.DataFrame(data)

    return df


@app.post("/probabilidad/")
def graf_probabilidad(edades: List[int]):
    df = calc_probabilidad(edades)
    data = [
        go.Table(
            header=dict(
                values=list(df.columns),
                fill_color="lightblue",
                align="center",
                font=dict(color="black", size=14,),
                line=dict(color='black', width=1),
                height=40,

            ),
            cells=dict(
                values=[df[col] for col in df.columns],
                fill=dict(color="lightgrey"),
                line=dict(color='black', width=1),
                align="center",
                font=dict(color="black", size=14, ),
                height=40,
            ),
        )
    ]

    num_filas = len(df)
    num_columnas = len(df.columns)
    ancho = num_columnas * 220
    altura = num_filas * 40

    layout = go.Layout(width=ancho, height=altura, autosize=False)
    fig = go.Figure(data=data, layout=layout)

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                      yaxis=dict(fixedrange=False, showgrid=False),)

    fig_bytes = pio.to_image(fig, format="png")
    fig_base64 = base64.b64encode(fig_bytes).decode("utf-8")

    return fig_base64
