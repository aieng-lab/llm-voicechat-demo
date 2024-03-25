# llm-voicechat-demo

# Linux installation

For this project you'll need python 3.9.18.

1. Clone this repository:
```
git clone https://github.com/aieng-lab/llm-voicechat-demo.git
```

2. Clone BreezeStyleSheets into Code directory:
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

