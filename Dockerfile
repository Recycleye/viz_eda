FROM python:3.7-slim

LABEL maintainer = "Ricky Ma <ricky.ma@alumni.ubc.ca>"


RUN apt-get update \
&& apt-get install gcc -y \
&& apt-get install libgtk2.0-dev -y \
&& apt-get clean

COPY requirements.txt /
RUN pip install numpy
RUN pip install -r /requirements.txt

COPY ./ ./
WORKDIR /src
EXPOSE 8050

#RUN mkdir /app
ENV PATH=$PATH:/src
ENV PYTHONPATH /src
#ADD . /src
#CMD ["python", "./src.py"]
ENTRYPOINT [ "python3" ]
CMD ["app.py"]
