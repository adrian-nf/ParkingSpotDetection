# Usar una imagen base de Python
FROM python:3.11

# Instalar ffmpeg y dependencias necesarias
RUN apt-get update && \
    apt-get install -y ffmpeg libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de requerimientos
COPY requirements.txt .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente y los modelos
COPY model_server.py .
COPY parking_spot_detection.py .
COPY models ./models

# Exponer el puerto en el que correrá el servidor
EXPOSE 8000

# Comando para ejecutar el servidor
CMD ["python", "model_server.py"]