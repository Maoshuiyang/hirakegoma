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
fps = 8.61328125 #9 #動画再生速度と音声とのmachingのために変更する

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
    frame1 = cv2.resize(frame, dsize=(1280, 960))
    
    # 書き込み
    video.write(frame)
            
    # キー入力待機
    if cv2.waitKey(97) & 0xFF == ord('q'):
        break
    # 画面表示
    cv2.imshow('frame', frame1)    
    output = stream.write(input)
    frames.append(input)
    if time.time()-st >= 5:
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
        length_seconds = base_sound.duration_seconds  # 長さを確認
        #print('sound_length_seconds={}'.format(length_seconds))
        
        cap = cv2.VideoCapture(outputfile)
        video_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) # フレーム数を取得する
        video_fps = cap.get(cv2.CAP_PROP_FPS)           # FPS を取得する
        video_len_sec = video_frame / video_fps         # 長さ（秒）を計算する
        videoFps= video_frame / length_seconds
        print('correct_videoFps={}'.format(videoFps))
        #print('sound_length_seconds={},video_len_sec={}'.format(length_seconds,video_len_sec))      # 長さ（秒）を出力する   
        
        clip_output = mp.VideoFileClip(outputfile).subclip()
        clip_output.write_videofile(outputfile.replace('.avi', '.mp4'), audio='out'+str(s)+'.mp3')
        dt_now = datetime.datetime.now()
        with open('./dog-cat/file.txt', 'a') as f: #w
            f.write(str(dt_now)+'_'+str(s)+' ; '+outputfile+'sound_length_seconds={},video_len_sec={}'.format(length_seconds,video_len_sec)+'\n')
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
