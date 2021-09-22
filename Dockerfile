FROM public.ecr.aws/bitnami/python:3.8.12

RUN mkdir my-model
WORKDIR /my-model

COPY requirements.txt requirements.txt 
RUN python3.8 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt 

COPY train.py ./train.py

CMD python3 train.py

