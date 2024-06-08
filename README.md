# llm-voicechat-demo

## Installation

For this project you'll need python 3.9.18.

1. Clone this repository:
```
git clone https://github.com/aieng-lab/llm-voicechat-demo.git
```

2. create a virtual environment:
   - Using conda:
   ```
   conda create --name voicebot python==3.9.18
   conda activate voicebot
   ```
   
   - Using Python virtualenv:
   ```
   python3.9.18 -m venv voicebot
   source voicebot/bin/activate
   ```
   - Or you can build a Docker instance:
     ```
     cd llm-voicechat-demo
     docker build . -t bot_backend -f DockerProject/Dockerfile
     docker run -p 5000:5000 --runtime=nvidia --gpus all bot_backend
     ```
     If you face problems with your GPU when using docker, refer to this question on Stackoverflow [click here](https://stackoverflow.com/questions/25185405/using-gpu-from-a-docker-container). 
     When you run tha last command, all needed components will be downloaded. 
     Models download progress doesn't show and takes a while, if you're worried that the program is stuck, watch your network traffic to make sure it's downloading.
   
3. Install the required libraries:
```
pip install -r requirements.txt
```

4. You need two terminal windows at llm-voicechat-demo/Code directory
    - In the first terminal run (if you're using Docker, you don't need to run the backend, because it's already running. Just run the GUI):
    ```
    python FlaskSocketIO_backend.py
    ```
    - In the second terminal run:
    ```
    python FlaskSocketIO_GUI.py
    ```


## How to use:

After executing the commands in Step 5, wait until all components are loaded.
(Even if the GUI appears, make sure that the FlaskAPI is running, because loading the models takes more time than loading the GUI).

//Link to GUI screenshot

In the middle of the screen a text field will inform you about the current state of the program:

   - "Ich schlafe ...": The program is initialized, and ready to be started.
   - "Ich spreche ...": The generated speech is being played and plotted.
   - "Ich höre zu ...": The program is ready to record a voice query.
   - "Ich überlege was ich antworte ...": The query is being processed.

Once everything is loaded, you can start the program by pressing on the "Starte Gespräch" button.

The program will not record until "Ich höre zu ..." is shown, that's when you can query.

"Beende Gespräch" button will reset all parameters and take the program back to "Ich shclafe ..." state.

Everything must be closed from terminals.



## Architicture:
The project is a Websocket server with RestAPI using python Flask-SocketIO with front- and backend.

- Frontend:
   - Holds GUI designed using PyQt5.
   - Holds SocketIO AsyncClient to communicate with the backend. <br /><br />
   - Controls both microphone and speaker to record an play audio.
   - Uses QRunnable to work as threads.
- Backend:
   - Holds The FlaskSocketIO AsyncServer to communicate with frontend.
   - Holds Speech-to-Text model.
   - Holds Text-to-Text model.
   - Holds Text-to-Speech model.
   - Runs the server on the main thread.
   - When a request is received, a new thread is created to preocess and send back response.

## How it works:

When "Starte Gespräch" is clicked, the frontend starts with playing a pre-recorded welcome message and connects to the backend through socketIO client.
Then the microphone is initialized and records user's query.

After an audio is recorded, it will be sent to the Flask-SocketIO server (backend) as GET request.

The backend receives the request on a Flask route, creates a new thread to process it as follows:

1. Transcribes the audio data using a speech-to-text model (WhisperLargeV2)
2. Streams generated text on the transcribtion using a large language model (FastChat/Vicuna).
3. Generates speech for each chunk of generated text using text-to-speech model (thorsten/vits from XTT2/Coqui)
4. Each generated speech would be sent back to frontend using socketIO as audio bytes.

The frontend reformat and normalize those bytes to be played and plotted synchronously.

At the end of generation the backend emits a signal to inform the frontend.

When frontend receives end-of-generation signal, it reactivates the microphone to record the next request.



