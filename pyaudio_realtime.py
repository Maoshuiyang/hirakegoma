# -*- coding:utf-8 -*-

import pyaudio
import time
import matplotlib.pyplot as plt
import numpy as np
import wave
from scipy.fftpack import fft, ifft
from scipy import signal
import os

def start_measure():
    CHUNK=1024
    RATE=44100 #11025 #22050  #44100
    p1=pyaudio.PyAudio()
    audio_index=1
    input1 = []
    stream1=p1.open(format = pyaudio.paInt16,
                  channels = 1,
                  rate = RATE,
                  input_device_index = audio_index,  
                  frames_per_buffer = CHUNK,
                  input = True) 
    input1 =stream1.read(CHUNK)
    sig1 = np.frombuffer(input1, dtype="int16")/32768.0
    while True:
        if max(sig1) > 0.0002:
            break
        input1 =stream1.read(CHUNK)
        sig1 = np.frombuffer(input1, dtype="int16")/32768.0
    stream1.stop_stream()
    stream1.close()
    p1.terminate()
    return

fig = plt.figure(figsize=(6, 5))
ax2 = fig.add_subplot(111)
N=50
CHUNK=1024*N
RATE=44100 #11025 #22050  #44100
p=pyaudio.PyAudio()
fr = RATE
fn=51200*N/50
fs=fn/fr
path='./dog-cat/3karasu/wav/'
FORMAT = pyaudio.paInt16
CHANNELS = 1  #monoral
#サンプリングレート、マイク性能に依存

for s in range(0,1000,1):
    print(s)
    start_measure()
    audio_index=1
    stream=p.open(format = pyaudio.paInt16,
              channels = 1,
              rate = RATE,
              input_device_index = audio_index,    
              frames_per_buffer = CHUNK,
              input = True)

    input = []
    start_time=time.time()
    input = stream.read(CHUNK)
    stop_time=time.time()
    frames = []
    frames.append(input)
    
    wf = wave.open(path+str(s)+'.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))  #width=2 ; 16bit
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    sig =[]
    sig = np.frombuffer(input, dtype="int16")  /32768.0

    nperseg = 1024
    f, t, Zxx = signal.stft(sig, fs=fn, nperseg=nperseg)
    ax2.pcolormesh(fs*t, f/fs/2, np.abs(Zxx), cmap='hsv')
    ax2.set_xlim(0,fs)
    ax2.set_ylim(20,20000)
    ax2.set_yscale('log')
    ax2.set_axis_off()
    
    plt.pause(0.01)
    plt.savefig('out_test/figure.jpg') #out_test/figure.jpg  #'train_images/0/figure' +str(s)+'.jpg' #dog-cat/1/figure' +str(s)+'.jpg
    #output = stream.write(input)
    #plt.savefig(path+'figure'+str(s)+'.jpg')

stream.stop_stream()
stream.close()
p.terminate()

print( "Stop Streaming")