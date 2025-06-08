FROM python:3.10-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto de Flask
EXPOSE 5000

# Ejecutar el servidor directamente (no usar flask run)
CMD ["python", "app.py"]
