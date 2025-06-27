@echo off
pyinstaller --add-data="C:\Program Files\Python39\Lib\site-packages\facenet_pytorch\data;facenet_pytorch\data" --noconsole MindsEye.py
pause