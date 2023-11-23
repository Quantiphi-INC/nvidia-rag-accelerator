FROM python:3.10

WORKDIR /frontend

COPY requirements.txt /frontend/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /frontend/requirements.txt

COPY ./ /frontend/

EXPOSE 8501

CMD ["streamlit", "run", "app.py","--server.port","8501"]