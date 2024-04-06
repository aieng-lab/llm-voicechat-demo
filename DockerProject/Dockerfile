# Dockerfile
FROM python:3.9

# FROM ubuntu:latest
# WORKDIR /code

# RUN apt-get update && apt-get autoclean

# RUN apt-get update && apt-get install -y --no-install-recommends python3-pip

# COPY requirements.txt /code/
ADD requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

# COPY ./welcome_message_2.wav /code/
ADD welcome_message_2.wav .
# COPY ./Models.py /code/
ADD Models.py .
# COPY ./FlaskSocketIO_backend.py /code/
ADD FlaskSocketIO_backend.py .
# COPY ./main.ui /code/
ADD main.ui .
# COPY ./FlaskSocketIO_GUI.py /code/
ADD FlaskSocketIO_GUI.py .

RUN apt-get update
RUN apt-get install libasound-dev libportaudio2 libportaudiocpp0 portaudio19-dev -y
RUN pip install pyaudio==0.2.13
RUN apt-get install ffmpeg -y

# COPY ./script.sh /code/
ADD script.sh .
RUN chmod a+x script.sh

ADD BreezeStyleSheets ./BreezeStyleSheets
ENV DISPLAY="0"
# ENV QT_DEBUG_PLUGINS=1
# ENV QT_QPA_PLATFORM=offscreen
# RUN apt-get install -y '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
# RUN apt-get install libxcb-xinerama0

# RUN apt-get install libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev libxcb-shape0-dev libxcb-xkb-dev
# RUN apt-get install libxkbcommon-x11-dev
# RUN apt-get install -y libxcb-util-dev
# RUN apt-get -y install build-essential openssl libssl-dev libssl1.0 libgl1-mesa-dev libqt5x11extras5 '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
RUN apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev -y
# CMD [ "python", "-u", "FlaskSocketIO_backend.py" ]

RUN apt-get install -y portaudio19-dev pulseaudio
EXPOSE 4713

CMD [ "./script.sh" ]


