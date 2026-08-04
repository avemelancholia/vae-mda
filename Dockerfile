FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt /app/
RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt
# Optional: ClearML online logging, auto-discovered from .env at runtime.
# Drop this line for a lean image that only logs locally to outputs/.
RUN python -m pip install --no-cache-dir clearml==1.13.1

COPY . /app/

ENTRYPOINT ["./entrypoint.sh"]
