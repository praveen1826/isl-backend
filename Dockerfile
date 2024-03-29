FROM python:3.11-slim-bullseye

RUN mkdir -p /home/isl-backend

WORKDIR /home/isl-backend

COPY . .

RUN apt update

RUN pip install tensorflow --no-cache-dir

RUN apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6

RUN pip3 install django Pillow django-sslserver opencv-python  tensorflow_hub scikit-learn django-cors-headers 

CMD ["python" , "manage.py" , "runserver" , "0.0.0.0:8000"]