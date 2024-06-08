# Dockerfile
FROM python:3.9

ADD requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

ADD Code/welcome_message.wav .
ADD Code/Models.py .
ADD Code/FlaskSocketIO_backend.py .
ADD Code/params.json .
RUN apt-get update
RUN apt-get install libasound-dev libportaudio2 libportaudiocpp0 portaudio19-dev -y
RUN pip install pyaudio==0.2.13
RUN apt-get install ffmpeg -y

RUN apt-get install -y portaudio19-dev pulseaudio

CMD [ "python", "FlaskSocketIO_backend.py" ]


