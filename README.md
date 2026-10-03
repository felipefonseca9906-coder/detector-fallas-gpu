# Detector de fallas de GPU

Servicio de machine learning que clasifica ventanas de telemetría de GPU en uno de cuatro estados: `normal`, `sobrecalentamiento`, `degradacion_memoria` o `falla_alimentacion`. El modelo se expone mediante una API REST con FastAPI y puede ejecutarse dentro de Docker.

## Estructura del proyecto

```text
detector-fallas-gpu/
|- src/
|  |- features.py      # cálculo de features
|  |- schema.py        # contrato de datos con Pandera
|  |- train.py         # entrenamiento y serialización
|  |- api.py           # API de FastAPI
|- models/
|  |- modelo.joblib    # modelo serializado
|- Dockerfile          # configuración de la imagen Docker
|- .dockerignore       # archivos excluidos de Docker
|- requirements.txt    # dependencias del proyecto
|- README.md           # instrucciones del proyecto
```

El dataset de trabajo se encuentra en `data/telemetria_publica.csv`.

## Flujo del modelo

El dataset contiene una lectura por segundo. El entrenamiento agrupa las lecturas por `episodio_id` y las divide en ventanas de 30 segundos, sin cruzar episodios. Cada ventana produce 22 features: media, desviación estándar, mínimo y máximo de temperatura, potencia, utilización, reloj y errores ECC; además del total de errores ECC y el rango de potencia. La etiqueta `estado` se conserva aparte como objetivo; los identificadores no se usan como features.

Antes de calcular las features, cada ventana se valida con Pandera. El contrato exige temperatura entre 0 y 120 °C, potencia mayor que cero, utilización entre 0 y 100 %, reloj positivo, errores ECC no negativos y uno de los cuatro estados válidos. Si una lectura no cumple el contrato, se elimina esa lectura, pero se conserva su ventana para el entrenamiento cuando aún contiene lecturas válidas. El CSV proporcionado tiene tres lecturas con potencia no positiva; se eliminan esas tres lecturas y se conservan sus tres ventanas con 29 lecturas cada una.

El entrenamiento usa un `RandomForestClassifier` dentro de un pipeline de scikit-learn y guarda el resultado en `models/modelo.joblib`. Las dependencias están fijadas para mantener compatibilidad con el modelo serializado.

## Requisitos

- Python 3.12 para ejecutar localmente.
- Docker Desktop o Docker Engine para ejecutar el contenedor.

## Comandos para ejecutar localmente

Desde la carpeta raíz del proyecto, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000
```

El entrenamiento lee `data/telemetria_publica.csv` y sobrescribe `models/modelo.joblib`. Debe ejecutarse antes de iniciar la API si se quiere regenerar el modelo. Cuando Uvicorn esté corriendo, abre http://localhost:8000/docs para explorar la API.

Para detener el servidor local, presiona `Ctrl+C` en la terminal.

## API de predicción

`POST /predecir` recibe un objeto JSON con la propiedad `lecturas`, que contiene una lista de al menos 10 lecturas. Cada lectura debe incluir exactamente estos campos:

| Campo | Tipo esperado | Validación |
| --- | --- | --- |
| `temp_c` | Número decimal | Entre 0 y 120 °C, inclusive |
| `power_w` | Número decimal | Mayor que 0 |
| `util_pct` | Número decimal | Entre 0 y 100 %, inclusive |
| `clock_mhz` | Número decimal | Mayor que 0 |
| `ecc_errors` | Entero | Mayor o igual que 0 |

La validación es estricta: no se convierten automáticamente valores de tipos incompatibles. Tampoco se aceptan valores no finitos como `NaN` o infinito. No se permiten propiedades adicionales ni en el objeto de la solicitud ni en cada lectura. Para predecir no se envían `episodio_id`, `segundo` ni `estado`; esos campos se utilizan en el procesamiento del dataset de entrenamiento, no en la API.

Si falta un campo, hay campos adicionales, un valor no cumple su tipo o rango, o se envían menos de 10 lecturas, la API responde con HTTP `422` y no ejecuta el modelo. No hay un máximo de lecturas configurado. Cuando la solicitud es válida, se calculan las features agregadas de todas las lecturas recibidas y se devuelve la clase con mayor probabilidad junto con su confianza.

Ejemplo para PowerShell, que envía 10 lecturas:

```powershell
$lectura = @{ temp_c = 88.3; power_w = 296.5; util_pct = 95.8; clock_mhz = 2038.0; ecc_errors = 0 }
$lecturas = 1..10 | ForEach-Object { $lectura }
$cuerpo = @{ lecturas = $lecturas } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://localhost:8000/predecir -Method Post -ContentType application/json -Body $cuerpo
```

La respuesta incluye el estado predicho y la probabilidad asignada a la clase predicha:

```json
{
	"estado_predicho": "sobrecalentamiento",
	"confianza": 0.98
}
```

## Comandos para ejecutar con Docker

Desde la carpeta raíz, construye la imagen y ejecuta el contenedor:

```powershell
docker build -t detector-gpu .
docker run --rm -p 8000:8000 detector-gpu
```
Si la imagen ya esta creada, solo hay que correrla con docker abierto 

La imagen incluye el código y el modelo entrenado; no incluye el CSV ni vuelve a entrenar al iniciarse. Con el contenedor activo, la API queda disponible en http://localhost:8000 y su documentación interactiva en http://localhost:8000/docs. Detén el contenedor con `Ctrl+C`.
