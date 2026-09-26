# detector-fallas-gpu
Parcial MCDP 

## Construir la imagen

```bash
docker build -t detector-gpu .
```

## Ejecutar el contenedor

```bash
docker run --rm -p 8000:8000 detector-gpu
```

La documentación interactiva de la API queda disponible en http://localhost:8000/docs.
