FROM python:3.10-slim

WORKDIR /app

# system dependencies (IMPORTANT for YOLO + OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x start.sh

EXPOSE 8501

CMD ["bash", "start.sh"]