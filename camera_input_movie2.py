# -*- coding: utf-8 -*-
import cv2
import pyaudio
import sys
import time
import wave
import pydub
from pydub import AudioSegment
import moviepy.editor as mp
import datetime

cap = cv2.VideoCapture(0)
fps = 8 #動画再生速度と音声とのmachingのために変更する

# 録画する動画のフレームサイズ（webカメラと同じにする）
size = (640, 480)
s=0
st=time.time()
# 出力する動画ファイルの設定
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file='./dog-cat/output'
dt_now = datetime.datetime.now()
outputfile=output_file+str(st)+'.avi'
video = cv2.VideoWriter(outputfile, fourcc, fps, size)

CHUNK = 1024*5 #1024
FORMAT = pyaudio.paInt16
CHANNELS = 1  #monoral
#サンプリングレート、マイク性能に依存
RATE = 44100
p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            output = True)
frames = []

while True:
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
    if time.time()-st >= 60:
        # 終了処理
        cap.release()
        video.release()
        cv2.destroyAllWindows()
        
        s +=1
        wf = wave.open('out'+str(s)+'.wav', 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))  #width=2 ; 16bit
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()    
        
        base_sound = AudioSegment.from_file('out'+str(s)+'.wav', format="wav")
        base_sound.export('out'+str(s)+'.mp3', format="mp3")  # 保存する
                
        clip_output = mp.VideoFileClip(outputfile).subclip()
        clip_output.write_videofile(outputfile.replace('.avi', '.mp4'), audio='out'+str(s)+'.mp3')
        dt_now = datetime.datetime.now()
        with open('./dog-cat/file.txt', 'a') as f: #w
            f.write(str(dt_now)+'_'+str(s)+' ; '+outputfile+'\n')
        outputfile=output_file+str(st)+'.avi'
        cap = cv2.VideoCapture(0)
        video = cv2.VideoWriter(outputfile, fourcc, fps, size)
        ret, frame = cap.read()
        frames=[]
        st=time.time()
        
# 終了処理
cap.release()
video.release()
cv2.destroyAllWindows()
