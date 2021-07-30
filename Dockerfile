FROM python:3.8-slim-buster

WORKDIR /project

RUN python -m pip install --upgrade pip
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /project 

CMD [ "python", "./main.py" ]