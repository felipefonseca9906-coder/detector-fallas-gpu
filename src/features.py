import pandas as pd


def generar_ventanas(telemetria: pd.DataFrame, tamano_ventana: int = 30) -> list[pd.DataFrame]:
	if tamano_ventana < 1:
		raise ValueError("tamano_ventana debe ser mayor que cero")

	ventanas = []
	for _, episodio in telemetria.groupby("episodio_id", sort=False):
		episodio = episodio.sort_values("segundo")
		for inicio in range(0, len(episodio), tamano_ventana):
			ventanas.append(episodio.iloc[inicio:inicio + tamano_ventana].copy())

	return ventanas
