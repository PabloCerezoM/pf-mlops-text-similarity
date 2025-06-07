# Imagen base ligera con Python
FROM python:3.10-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements y archivo de entorno
COPY requirements.txt .
#COPY .env .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la app
COPY app/ app/
COPY tests/ tests/ 
# Cargar variables de entorno automáticamente
ENV PYTHONUNBUFFERED=1

# Exponer el puerto por defecto de FastAPI
EXPOSE 8000

# Comando para ejecutar la app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
