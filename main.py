from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from typing import List

app = FastAPI()

# Configurar el middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Link a la aplicación React
    allow_credentials=True,
    allow_methods=["*"],  # Esto permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Esto permite todos los encabezados
)

codigos_patologias = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
]

nombres_patologias = [
    "aborto",
    "af perinatales",
    "asma",
    "bronquitis/ bronquiolitis aguda",
    "causas obtétricas indirec",
    "chagas",
    "colelitiasis/colecistitis",
    "compl at medica/quirúrgica",
    "compl embarazo",
    "compl parto",
    "compl puerperio",
    "covid-19",
    "def nutricionales",
    "dengue",
    "diabetes",
    "ecv",
    "enf apéndice",
    "enf crónica vri",
    "enf hipertensivas",
    "enf hígado",
    "enf infecciosas intestinales",
    "enf oido/ ap mastoides",
    "enf ojos",
    "enf piel/ tcs",
    "enf páncreas",
    "enf sangre/ org hematopoyético",
    "enf sist osteomuscular",
    "enf sist urinario",
    "enteritis/ colitis no infecciosa",
    "envenenamiento/ tóxico",
    "epilepsia",
    "epoc",
    "fac que influyen en la salud/ contacto con serv salud",
    "hallazgos clínicos/ laboratorio anormales",
    "hepatitis virales",
    "hernias",
    "hiperplasia próstata",
    "iam",
    "influenza/ neumonía",
    "ira sup",
    "its",
    "leucemia",
    "malf congénitas",
    "meningitis bacteriana",
    "meningitis viral",
    "obesidad",
    "otras afec obstétricas",
    "otras causas externas",
    "otras enf cardíacas",
    "otras enf de vrs",
    "otras enf del peritoneo",
    "otras enf endócrinas/ metabólicas",
    "otras enf infecciosas/ parasitarias",
    "otras enf sist circulatorio",
    "otras enf sist digestivo",
    "otras enf sist genitourinario",
    "otras enf sist nervioso",
    "otras enf sist respiratorio",
    "otras ira inferiores",
    "otros trastornos mentales",
    "otros tumores malignos",
    "parotiditis infecciosa",
    "parto",
    "quemadura/ congelamiento",
    "salpingitis/ ooforitis",
    "sarampión",
    "sec causas ext",
    "septicemias",
    "t cabeza/ cuello",
    "t mmss y mmii",
    "t múltiples",
    "t tórax/ abdomen/ lumbopelvis",
    "tos ferina",
    "trast por alcohol",
    "trast tiroides",
    "tuberculosis",
    "tumor genitales fem",
    "tumor genitales masc",
    "tumor in situ/ benigno",
    "tumor mama",
    "tumor org digestivos",
    "tumor org intratoráxicos",
    "tumor org urinario",
    "tumor próstata",
    "tumor útero",
    "ulcera gástrica/ duodenal",
    "varicela",
    "vih",
]

# Cargar el modelo
modelo = joblib.load("./Regresion_pat_def.pkl")
probabilidad = joblib.load("./ProbAltaDefEdad.pkl")


@app.post("/predecir/")
def predecir_valores(codPatologias: List[int]):
    # Crear un dataframe con los valores recibidos
    features = pd.DataFrame({"causa_egreso": codPatologias})

    # Realizar la predicción
    prediccion = modelo.predict(features)

    # Crear una tabla con valores input y predicciones
    data = {"Enfermedad": codPatologias, "Predicción": prediccion}
    df = pd.DataFrame(data)

    df["Enfermedad"] = df["Enfermedad"].replace(codigos_patologias, nombres_patologias)

    # Calcular la cantidad de True para cada número de enfermedad
    true_counts = df.groupby("Enfermedad")["Predicción"].sum().reset_index()

    # Calcular el total de cada número de enfermedad
    total_counts = df["Enfermedad"].value_counts().reset_index()
    total_counts.columns = ["Enfermedad", "Total"]

    # Combinar ambas columnas
    result_df = pd.merge(true_counts, total_counts, on="Enfermedad", how="left")

    # Convertir la tabla en una imagen
    plt.figure(figsize=(9, 2), dpi=300)
    plt.axis("off")
    plt.table(
        cellText=result_df.values,
        colLabels=["Enfermedad", "Defunciones", "Casos"],
        cellLoc="center",
        loc="center",
        colColours=["lightblue", "yellow", "lightgreen"],
    )

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight", pad_inches=0.2)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()

    return img_base64

@app.post("/probabilidad/")
def predecir_egreso(edades: List[int]):
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
    data = {"Edad": edades,"Probabilidad Alta": probabilidad_alta, "Probabilidad Defunción": probabilidad_defuncion}
    df = pd.DataFrame(data)

        # Crear el gráfico de tabla
    fig, ax = plt.subplots(figsize=(9, 2), dpi=300)
    ax.axis('off')
    table_data = [df.columns] + df.values.tolist()
    table = ax.table(cellText=table_data,
                     cellLoc='center',
                     loc='center',
                     colColours=["lightblue"] * 3 + ["lightgreen"] * 3)

    # Estilo para la tipografía en negrita (bold) y celdas grises (grey)
    for i, cell in enumerate(table._cells.values()):
        cell.set_text_props(weight='bold')
        if i > 2:  # Saltar las primeras tres celdas que son encabezados
            cell.set_facecolor("lightgrey")

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight", pad_inches=0.2)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()

    return img_base64

