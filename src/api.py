from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.features import extraer_features


RUTA_MODELO = Path(__file__).resolve().parents[1] / "models" / "modelo.joblib"
modelo = joblib.load(RUTA_MODELO)

app = FastAPI(title="Detector de fallas de GPU")


class SolicitudPrediccion(BaseModel):
	lecturas: list[dict[str, Any]]


class RespuestaPrediccion(BaseModel):
	estado_predicho: str
	confianza: float


@app.post("/predecir", response_model=RespuestaPrediccion)
def predecir(solicitud: SolicitudPrediccion) -> RespuestaPrediccion:
	"""Calcula las features de una ventana y devuelve el estado predicho con su confianza."""
	ventana = pd.DataFrame(solicitud.lecturas)
	features = pd.DataFrame([extraer_features(ventana)])
	probabilidades = modelo.predict_proba(features)[0]
	indice_predicho = probabilidades.argmax()

	return RespuestaPrediccion(
		estado_predicho=str(modelo.classes_[indice_predicho]),
		confianza=float(probabilidades[indice_predicho]),
	)
