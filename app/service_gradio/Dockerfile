# Usar una imagen base de Python
FROM python:3.11

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de requerimientos
COPY requirements.txt .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY app.py .

# Exponer el puerto en el que correrá Gradio
EXPOSE 7860

# Comando para ejecutar la aplicación Gradio
CMD ["python", "app.py"]