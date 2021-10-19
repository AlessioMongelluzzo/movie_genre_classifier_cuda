FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04
CMD nvidia-smi

WORKDIR /app
# install pip and basic py dependencies
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y apt-utils \
curl unzip python3.8 python3-distutils python3-apt
# pip3.8
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py
# app packages
RUN python3.8 -m pip install tensorflow==2.6.0 numpy==1.19.5 gunicorn==20.1.0 flask==2.0.1

ADD ./models ./models
ADD ./aux ./aux
ADD gen_clf_server.py gen_clf_server.py

EXPOSE 5000
CMD python3.8 -m gunicorn --bind 0.0.0.0:5000 gen_clf_server:app
#CMD python3.8