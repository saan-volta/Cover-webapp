FROM python:3.11-slim

WORKDIR  /src

RUN pip install --no-cache-dir "numpy<2"
RUN pip install --no-cache-dir "torch==2.2.2" --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app/
COPY Cover ./Cover/
COPY static ./static/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host" , "0.0.0.0", "--port", "8000"]

