from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import base64
import numpy as np
import torch
import pandas as pd  # Agregamos la importación necesaria para pandas

app = FastAPI()

class ImageInput(BaseModel):
    imagen_base64: str

@app.post("/evaluar_defectos/")
def evaluar_defectos(image_input: ImageInput):
    try:
        # Decodificar la imagen base64
        imagen_bytes = base64.b64decode(image_input.imagen_base64)
        imagen_np = np.frombuffer(imagen_bytes, dtype=np.uint8)
        img = cv2.imdecode(imagen_np, cv2.IMREAD_COLOR)

        # Cargar el modelo (reemplaza esta parte con la lógica de carga de tu modelo)
        model_path = 'best.pt'
        model = torch.hub.load('./', 'custom', model_path, source='local', force_reload=True)  # Puede dar error

        model.eval()

        # Ejecutar el modelo en la imagen
        # Cambia esta parte según los requisitos de entrada de tu modelo
        results = model(img)

        # Procesar los resultados
        # Suponiendo que el modelo devuelve un DataFrame de pandas con las detecciones
        data = results.pandas().xyxy[0]

        # Convertir a formato JSON y devolver como resultado
        result_json = data.to_json(orient='records')
        return {
            'statusCode': 200,
            'body': result_json
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/evaluar_defectos_oscuros/")
def evaluar_defectos_oscuros(image_input: ImageInput):
    try:
        # Decodificar la imagen base64
        imagen_bytes = base64.b64decode(image_input.imagen_base64)
        imagen_np = np.frombuffer(imagen_bytes, dtype=np.uint8)
        img = cv2.imdecode(imagen_np, cv2.IMREAD_COLOR)

        # Cargar el modelo (reemplaza esta parte con la lógica de carga de tu modelo)
        model_path = 'best.pt'
        model = torch.hub.load('./', 'custom', model_path, source='local', force_reload=True)  # Puede dar error

        model.eval()

        # Ejecutar el modelo en la imagen
        # Cambia esta parte según los requisitos de entrada de tu modelo
        results = model(img)

        # Procesar los resultados
        # Suponiendo que el modelo devuelve un DataFrame de pandas con las detecciones
        data = results.pandas().xyxy[0]

        # Convertir a formato JSON y devolver como resultado
        result_json = data.to_json(orient='records')
        return {
            'statusCode': 200,
            'body': result_json
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/evaluar_defectos_persona/")
def evaluar_defectos_persona(image_input: ImageInput):
    try:
        # Decodificar la imagen base64
        imagen_bytes = base64.b64decode(image_input.imagen_base64)
        imagen_np = np.frombuffer(imagen_bytes, dtype=np.uint8)
        img = cv2.imdecode(imagen_np, cv2.IMREAD_COLOR)

        # Cargar el modelo (reemplaza esta parte con la lógica de carga de tu modelo)
        model_path = 'best_236_160.pt'
        model = torch.hub.load('./', 'custom', model_path, source='local', force_reload=True)  # Puede dar error

        model.eval()

        # Ejecutar el modelo en la imagen
        # Cambia esta parte según los requisitos de entrada de tu modelo
        results = model(img)

        # Procesar los resultados
        # Suponiendo que el modelo devuelve un DataFrame de pandas con las detecciones
        data = results.pandas().xyxy[0]

        # Convertir a formato JSON y devolver como resultado
        result_json = data.to_json(orient='records')
        return {
            'statusCode': 200,
            'body': result_json
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
