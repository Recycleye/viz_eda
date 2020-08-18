FROM python:3.7-slim

LABEL maintainer = "Ricky Ma <ricky.ma@alumni.ubc.ca>"


RUN apt-get update \
&& apt-get install gcc -y \
&& apt-get install libgtk2.0-dev -y \
&& apt-get clean

COPY requirements.txt /
RUN pip install numpy
RUN pip install -r /requirements.txt

#COPY ./app /app
#WORKDIR "/app"
#EXPOSE 8050
#
#ENTRYPOINT [ "python3" ]
#CMD ["application.py"]

RUN mkdir /app
WORKDIR /app
ADD . /app/

ENTRYPOINT [ "python" ]
CMD ["application.py"]
