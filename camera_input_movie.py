# -*- coding: utf-8 -*-
import cv2
import pyaudio
import sys
import time
import wave
import pydub
from pydub import AudioSegment
import moviepy.editor as mp

cap = cv2.VideoCapture(0)
fps = 30

# 録画する動画のフレームサイズ（webカメラと同じにする）
size = (640, 480)

# 出力する動画ファイルの設定
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file='output.avi'
video = cv2.VideoWriter(output_file, fourcc, fps, size)

CHUNK = 1024*5 #1024
FORMAT = pyaudio.paInt16
CHANNELS = 1  #monoral
#サンプリングレート、マイク性能に依存
RATE = 22050  #44100
p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            output = True)
frames = []
while (cap.isOpened()):
    ret, frame = cap.read()
    input = stream.read(CHUNK)

    # 書き込み
    video.write(frame)
            
    # キー入力待機
    if cv2.waitKey(97) & 0xFF == ord('q'):
        break
    # 画面表示
    cv2.imshow('frame', frame)    
    output = stream.write(input)
    frames.append(input)

# 終了処理
cap.release()
video.release()
cv2.destroyAllWindows()

wf = wave.open('out.wav', 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))  #width=2 ; 16bit
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()    

base_sound = AudioSegment.from_file('out.wav', format="wav")
base_sound.export("out.mp3", format="mp3")  # 保存する

clip_output = mp.VideoFileClip(output_file).subclip()
clip_output.write_videofile(output_file.replace('.avi', '.mp4'), audio='out.mp3')


