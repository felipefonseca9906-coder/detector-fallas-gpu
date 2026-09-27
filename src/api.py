from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.features import extraer_features


RUTA_MODELO = Path(__file__).resolve().parents[1] / "models" / "modelo.joblib"
modelo = joblib.load(RUTA_MODELO)

app = FastAPI(title="Detector de fallas de GPU")


class LecturaTelemetria(BaseModel):
	model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

	temp_c: float = Field(ge=0, le=120)
	power_w: float = Field(gt=0)
	util_pct: float = Field(ge=0, le=100)
	clock_mhz: float = Field(gt=0)
	ecc_errors: int = Field(ge=0)


class SolicitudPrediccion(BaseModel):
	model_config = ConfigDict(strict=True, extra="forbid")

	lecturas: list[LecturaTelemetria] = Field(min_length=10)


class RespuestaPrediccion(BaseModel):
	estado_predicho: str
	confianza: float


@app.post("/predecir", response_model=RespuestaPrediccion)
def predecir(solicitud: SolicitudPrediccion) -> RespuestaPrediccion:
	"""Calcula las features de una ventana y devuelve el estado predicho con su confianza."""
	ventana = pd.DataFrame([lectura.model_dump() for lectura in solicitud.lecturas])
	features = pd.DataFrame([extraer_features(ventana)])
	probabilidades = modelo.predict_proba(features)[0]
	indice_predicho = probabilidades.argmax()

	return RespuestaPrediccion(
		estado_predicho=str(modelo.classes_[indice_predicho]),
		confianza=float(probabilidades[indice_predicho]),
	)
