FROM python:3.11-slim-bullseye

RUN mkdir -p /home/isl-backend

WORKDIR /home/isl-backend

COPY . .

RUN apt update

RUN pip install tensorflow --no-cache-dir

RUN pip3 install django Pillow django-sslserver opencv-python  tensorflow_hub scikit-learn django-cors-headers 

CMD ["python" , "manage.py" , "runserver" , "0.0.0.0:8000"]