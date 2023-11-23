FROM nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04

WORKDIR /backend
RUN apt update -y  && apt upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt install python3.10-full -y
RUN apt install -y python3-pip
COPY requirements.txt /backend/requirements.txt

RUN pip3 install --no-cache-dir --upgrade -r /backend/requirements.txt

COPY ./ /backend/

EXPOSE 9988

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9988"]