# llm-voicechat-demo

For this project you'll need python 3.9.18.

1. Clone this repository:
```
git clone https://github.com/aieng-lab/llm-voicechat-demo.git
```

3. Clone BreezeStyleSheets into project directory:
```
cd Code
git clone https://github.com/Alexhuszagh/BreezeStyleSheets.git
```

5. create a virtual enviroment:
   A. Using conda:
   ```
   conda create --name voicebot python==3.9.18
   conda activate voicebot
   ```
   
   B. Using Python vitualenv: 
   ```
   python3.9.18 -m venv voiceboty
   ```
   
7. Build BreezeStyleSheets:
   ```
   cd BreezeStyleSheets
   python configure.py --compiled-resource breeze_resources.py
   cd ..
   ```
11. Install the required libraries:
```
pip install -r requirements.txt
```
13. You need two terminal windows
    In the first terminal run:
    ```
    python FlaskSocketIO_backend.py
    ```
    In the second terminal run:
    ```
    python FlaskSocketIO_GUI.py
    ```

