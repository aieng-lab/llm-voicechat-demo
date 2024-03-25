# llm-voicechat-demo

For this project you'll need python 3.9.18.

1. Clone this repository:
'''
git clone https://github.com/aieng-lab/llm-voicechat-demo.git
'''
3. cd Code
4. Git BreezeStyleSheets: git clone https://github.com/Alexhuszagh/BreezeStyleSheets.git

5. create a virtual enviroment:
   A. Using conda:
       - conda create --name voicebot python==3.9.18
       - conda activate voicebot
   
   B. Using Python vitualenv: python3.9.18 -m venv voiceboty
   
6. cd BreezeStyleSheets
7. python configure.py --compiled-resource breeze_resources.py
8. cd ..
9. pip install -r requirements.txt
10. python FlaskSocketIO_backend.py
11. In a different terminal: python FlaskSocketIO_GUI.py

