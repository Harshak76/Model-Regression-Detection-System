FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "src/cli.py"]
CMD ["run", "--prompt", "prompts/v1.0.yaml", "--mock"]
