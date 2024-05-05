# llm-voicechat-demo

## Installation

For this project you'll need python 3.9.18.

1. Clone this repository:
```
git clone https://github.com/aieng-lab/llm-voicechat-demo.git
```

2. Clone BreezeStyleSheets into llm-voicechat-demo/Code/ directory:
```
cd llm-voicechat-demo/Code
git clone https://github.com/Alexhuszagh/BreezeStyleSheets.git
cd ..
```

3. create a virtual environment:
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
   
4. Install the required libraries:
```
pip install -r requirements.txt
```

5. You need two terminal windows at llm-voicechat-demo/Code directory
    - In the first terminal run:
    ```
    python FlaskSocketIO_backend.py
    ```
    - In the second terminal run:
    ```
    python FlaskSocketIO_GUI.py
    ```

If BreezeStyleSheets doesn't work, then it needs to be configured:
   ```
   cd BreezeStyleSheets
   python configure.py --compiled-resource breeze_resources.py
   cd ..
   ```
Then run 5.


## How to use:

After executing the commands in Step 5, wait until all components are loaded.
(Even if the GUI appears, make sure that the FlaskAPI is running, because loading the models takes more time than loading the GUI).

//Link to GUI screenshot

"Bot Status" field will show you the current state of the program:

   - "Ich schlafe ...": The program is initialized, and ready to be started.
   - "Ich spreche ...": The generated speech is being played and plotted.
   - "Ich höre zu ...": The program is ready to record a voice query.
   - "Ich überlege was ich antworte ...": The query is being processed.

Once everything is loaded, you can start the program by pressing on the "Starte Gespräch" button.

The program will not record until "Ich höre zu ..." is shown, that's when you can query.

"Beende Gespräch" button will reset all parameters and take the program back to "Ich shclafe ..." state.

Everything must be closed from the terminals.



## Architicture:
The project is developed using python Flask-SocketIO with front- and backend.

- The frontend consists of:
   1. The GUI designed using PyQt5.
   2. The SocketIO AsyncClient to communicate with the backend.
   It also controls both microphone and speaker to record an play audio.

- The backend consists of:
   1. The FlaskSocketIO AsyncServer to communicate with frontend.
   2. Speech-to-Text model.
   3. Text-to-Text model.
   4. Text-to-Speech model.

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



