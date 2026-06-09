FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 7860
CMD ["whoop-analytics", "web", "--port", "7860"]
