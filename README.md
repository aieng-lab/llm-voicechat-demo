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

   "INITIALIZING": The program is still loading.
   "IDLE": The program is initialized, and ready to be started.
   "SPEAKING": The generated speech is being played and plotted.
   "WAITING": Waiting on the microphone to record.
   "LISTENING": The program is ready to record a voice query.
   "PROCESSING": The query is being processed.

Once everything is loaded "IDLE", you can start the program by pressing on the "Start" button.

The program will not record any sound until "LISTENING" is shown, that's when you can query.

"Stop" button will prevent next steps, but will not terminate the current step of execution.

"Reset" button will reset all parameters and take the program back to "IDLE" state.

Everything must be closed from the terminals (Still working on closing).



## Architicture:
\\ Describe the architicture here

## How it works:

The GUI will record, play and plot audio data.

When an audio is recorded, it will be sent to the Flask-SocketIO server (backend) as a request.

When the backend receives a request, it will process it as follows:

1. Transcribes the audio data using a speech-to-text model (WhisperLargeV2)
2. Streams generated text using a large language model (FastChat/Vicuna).
3. Generates speech for each chunk of generated text using text-to-speech model (thorsten/vits from XTT2/Coquit)



