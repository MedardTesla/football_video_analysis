# Deux images depuis un seul fichier : l'API n'a pas besoin de CUDA.
#
#   docker build --target api    -t analyse-api .
#   docker build --target worker -t analyse-worker .
#
# L'API pèse quelques dizaines de mégaoctets, le worker plusieurs gigaoctets.
# Les séparer permet de faire tourner l'API en permanence sur une petite
# machine et de n'allumer le GPU que quand la file n'est pas vide.

# ---------------------------------------------------------------- API
FROM python:3.12-slim AS api

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# football_analysis/report.py est importé par le worker seulement, mais le
# paquet doit être présent : service/ en dépend pour les types.
COPY service/ ./service/
COPY football_analysis/ ./football_analysis/

# Utilisateur non privilégié : le service reçoit des fichiers d'inconnus.
RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

ENV FA_DATA_ROOT=/data
EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/sante').status==200 else 1)"
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ------------------------------------------------------------- WORKER
# Image CUDA : le pipeline est inutilisable sans GPU — mesuré à environ
# 3 vignettes par seconde sur processeur, soit des heures par match.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime AS worker

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# libGL et libglib : OpenCV les réclame même en version sans interface.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY service/ ./service/
COPY football_analysis/ ./football_analysis/

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

# Les poids sont montés, pas copiés : ils changent à chaque réentraînement et
# pèsent plus lourd que le code.
ENV FA_DATA_ROOT=/data
CMD ["python", "-m", "service.run_worker"]
