from pathlib import Path

import joblib
import pandas as pd
import pandera.pandas as pa
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.features import extraer_features, generar_ventanas
from src.schema import validar_telemetria


DIRECTORIO_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_DATOS = DIRECTORIO_PROYECTO / "data" / "telemetria_publica.csv"
RUTA_MODELO = DIRECTORIO_PROYECTO / "models" / "modelo.joblib"


def construir_dataset_valido(
	telemetria: pd.DataFrame, tamano_ventana: int = 30
) -> tuple[pd.DataFrame, pd.Series, int]:
	"""Extrae X e y de ventanas válidas y cuenta las ventanas rechazadas por el contrato."""
	filas_features = []
	etiquetas = []
	ventanas_rechazadas = 0

	for ventana in generar_ventanas(telemetria, tamano_ventana):
		try:
			validar_telemetria(ventana)
		except pa.errors.SchemaErrors:
			ventanas_rechazadas += 1
			continue

		estados = ventana["estado"].dropna().unique()
		if len(estados) != 1:
			raise ValueError("Cada ventana debe contener exactamente un estado")

		filas_features.append(extraer_features(ventana))
		etiquetas.append(estados[0])

	if not filas_features:
		raise ValueError("No hay ventanas válidas para entrenar")

	return (
		pd.DataFrame(filas_features),
		pd.Series(etiquetas, name="estado"),
		ventanas_rechazadas,
	)


def entrenar_modelo(
	ruta_datos: Path = RUTA_DATOS, ruta_modelo: Path = RUTA_MODELO
) -> tuple[Pipeline, int]:
	"""Entrena y serializa el pipeline usando únicamente ventanas que pasan el contrato."""
	telemetria = pd.read_csv(ruta_datos)
	X, y, ventanas_rechazadas = construir_dataset_valido(telemetria)
	modelo = Pipeline(
		[
			(
				"clasificador",
				RandomForestClassifier(
					n_estimators=300,
					class_weight="balanced",
					random_state=42,
					n_jobs=-1,
				),
			)
		]
	)
	modelo.fit(X, y)

	ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(modelo, ruta_modelo)
	return modelo, ventanas_rechazadas


if __name__ == "__main__":
	_, ventanas_rechazadas = entrenar_modelo()
	print(f"Modelo guardado en {RUTA_MODELO}; ventanas rechazadas: {ventanas_rechazadas}")
