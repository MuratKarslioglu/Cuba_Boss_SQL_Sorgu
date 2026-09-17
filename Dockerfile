# Hibrit kurulum kararinin Docker ayagi.
#
# ONEMLI: Docker Desktop macOS'ta bir Linux VM'i icinde calisir ve Apple'in
# Metal/MPS backend'ine erisemez. Bu nedenle EGITIM ve BENCHMARK bu imajda
# CALISTIRILMAZ -- spec §23'un latency/bellek olcumleri native ortamda alinir.
#
# Bu imajin amaci:
#   - deterministik test/dogrulama ortami
#   - Faz 6'da FastAPI servisinin tasiyicisi
#   - isverenin Linux sunucusuna tasima yolu
#
# Faz 6'da tek degisiklik CMD satiridir:
#   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# Faz 1+2 yalnizca cekirdek bagimliliklari gerektirir (torch yok, ~10 MB).
# Faz 6'ya gecerken burasi requirements.txt olur.
COPY requirements-core.txt ./
RUN pip install --no-cache-dir -r requirements-core.txt pytest

COPY pyproject.toml ./
COPY app ./app
COPY training ./training
COPY scripts ./scripts
COPY tests ./tests

CMD ["pytest"]
