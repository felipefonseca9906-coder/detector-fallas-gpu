import pandas as pd


def generar_ventanas(telemetria: pd.DataFrame, tamano_ventana: int = 30) -> list[pd.DataFrame]:
	"""Divide la telemetría en ventanas cronológicas sin mezclar episodios."""
	if tamano_ventana < 1:
		raise ValueError("tamano_ventana debe ser mayor que cero")

	ventanas = []
	for _, episodio in telemetria.groupby("episodio_id", sort=False):
		episodio = episodio.sort_values("segundo")
		for inicio in range(0, len(episodio), tamano_ventana):
			ventanas.append(episodio.iloc[inicio:inicio + tamano_ventana].copy())

	return ventanas


def extraer_features(ventana: pd.DataFrame) -> dict[str, float]:
	"""Calcula estadísticas de telemetría para una ventana, sin incluir etiquetas ni identificadores."""
	if ventana.empty:
		raise ValueError("La ventana no puede estar vacía")

	columnas_senal = ["temp_c", "power_w", "util_pct", "clock_mhz", "ecc_errors"]
	features = {}
	for columna in columnas_senal:
		valores = ventana[columna]
		features[f"{columna}_mean"] = float(valores.mean())
		features[f"{columna}_std"] = float(valores.std())
		features[f"{columna}_min"] = float(valores.min())
		features[f"{columna}_max"] = float(valores.max())

	features["ecc_errors_total"] = float(ventana["ecc_errors"].sum())
	features["power_w_range"] = float(ventana["power_w"].max() - ventana["power_w"].min())
	return features


def construir_dataset_entrenamiento(
	telemetria: pd.DataFrame, tamano_ventana: int = 30
) -> tuple[pd.DataFrame, pd.Series]:
	"""Construye X e y con las features y el estado correspondiente a cada ventana."""
	filas_features = []
	etiquetas = []
	for ventana in generar_ventanas(telemetria, tamano_ventana):
		estados = ventana["estado"].dropna().unique()
		if len(estados) != 1:
			raise ValueError("Cada ventana debe contener exactamente un estado")

		filas_features.append(extraer_features(ventana))
		etiquetas.append(estados[0])

	if not filas_features:
		raise ValueError("La telemetría no contiene ventanas para entrenar")

	return pd.DataFrame(filas_features), pd.Series(etiquetas, name="estado")
