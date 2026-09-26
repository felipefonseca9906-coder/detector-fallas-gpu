import pandas as pd
import pandera.pandas as pa


ESTADOS_VALIDOS = [
	"normal",
	"sobrecalentamiento",
	"degradacion_memoria",
	"falla_alimentacion",
]

TELEMETRIA_SCHEMA = pa.DataFrameSchema(
	{
		"episodio_id": pa.Column(int, nullable=False),
		"segundo": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0), nullable=False),
		"temp_c": pa.Column(float, checks=pa.Check.in_range(0, 120), nullable=False),
		"power_w": pa.Column(float, checks=pa.Check.greater_than(0), nullable=False),
		"util_pct": pa.Column(float, checks=pa.Check.in_range(0, 100), nullable=False),
		"clock_mhz": pa.Column(float, checks=pa.Check.greater_than(0), nullable=False),
		"ecc_errors": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0), nullable=False),
		"estado": pa.Column(str, checks=pa.Check.isin(ESTADOS_VALIDOS), nullable=False),
	},
	strict=True,
)


def validar_telemetria(telemetria: pd.DataFrame) -> pd.DataFrame:
	"""Valida columnas y rangos de telemetría antes de permitir que entren al pipeline."""
	return TELEMETRIA_SCHEMA.validate(telemetria, lazy=True)
