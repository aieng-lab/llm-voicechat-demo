# llm-voicechat-demo

For this project you'll need python 3.9.18.

1. Clone this repository:
```
git clone https://github.com/aieng-lab/llm-voicechat-demo.git
```

2. Clone BreezeStyleSheets into project directory:
```
cd Code
git clone https://github.com/Alexhuszagh/BreezeStyleSheets.git
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
   ```
   
5. Install the required libraries:
```
pip install -r requirements.txt
```
6. You need two terminal windows at llm-voicechat-demo/Code
    - In the first terminal run:
    ```
    python FlaskSocketIO_backend.py
    ```
    - In the second terminal run:
    ```
    python FlaskSocketIO_GUI.py
    ```

If BreezeStyleSheets doesn't work, then you need to setup its configuration:
   ```
   cd BreezeStyleSheets
   python configure.py --compiled-resource breeze_resources.py
   cd ..
   ```
Then run 6.

