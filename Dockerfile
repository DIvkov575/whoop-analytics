FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 7860
CMD ["uvicorn", "whoop_analytics.web.app:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "120"]
