FROM python:3.12-slim
WORKDIR /app
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt || true
COPY . .
ENV PORT=8080
CMD ["streamlit", "run", "app_cont.py", "--server.port=8080", "--server.address=0.0.0.0"]




