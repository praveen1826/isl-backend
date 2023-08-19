FROM python:3.11.3-alpine

RUN mkdir -p /home/islBackendT2I

WORKDIR /home/isl-backend

COPY . .

RUN apk update

RUN pipenv shell

RUN pip install django Pillow django-sslserver tensorflow opencv-python tensorflow_hub scikit-learn  

CMD ["python" , "manage.py" , "runserver" , "0.0.0.0:8000"]