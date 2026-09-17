FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libglib2.0-0 \
        libgl1 \
        git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src
COPY cpp ./cpp
COPY app ./app
COPY scripts ./scripts

RUN PYBIND11_DIR=$(python -m pybind11 --cmakedir) && \
    cmake -S cpp/pathox_native \
          -B cpp/pathox_native/build \
          -DCMAKE_PREFIX_PATH="$PYBIND11_DIR" && \
    cmake --build cpp/pathox_native/build \
          --config Release \
          -j2 && \
    mkdir -p src/pathox/native && \
    cp cpp/pathox_native/build/pathox_native*.so \
       src/pathox/native/

ENV PYTHONPATH=/app/src

EXPOSE 8501

CMD ["streamlit", "run", "app/pathox_dashboard.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]
