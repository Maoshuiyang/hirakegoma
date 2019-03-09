# -*- coding:utf-8 -*-

import pyaudio
import time
import matplotlib.pyplot as plt
import numpy as np
import wave
from scipy.fftpack import fft, ifft
from scipy import signal

def start_measure():
    CHUNK=1024
    RATE=44100 #11025 #22050  #44100
    p=pyaudio.PyAudio()
    input = []
    stream=p.open(format = pyaudio.paInt16,
                  channels = 1,
                  rate = RATE,
                  frames_per_buffer = CHUNK,
                  input = True) 
    input =stream.read(CHUNK)
    sig1 = np.frombuffer(input, dtype="int16")/32768.0
    while True:
        if max(sig1) > 0.001:
            break
        input =stream.read(CHUNK)
        sig1 = np.frombuffer(input, dtype="int16")/32768.0

    stream.stop_stream()
    stream.close()
    p.terminate()
    return

N=50
CHUNK=1024*N
RATE=44100 #11025 #22050  #44100
p=pyaudio.PyAudio()

stream=p.open(format = pyaudio.paInt16,
              channels = 1,
              rate = RATE,
              frames_per_buffer = CHUNK,
              input = True)

fig = plt.figure(figsize=(6, 5))
"""
ax1 = fig.add_subplot(311)
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel('Signal')
"""
ax2 = fig.add_subplot(111)
#ax2.set_ylabel('Freq[Hz]')
#ax2.set_xlabel('Time [sec]')
"""
ax3 = fig.add_subplot(313)
ax3.set_xlabel('Freq[Hz]')
ax3.set_xscale('log')
ax3.set_ylabel('Power')
"""
start=time.time()
stop_time=time.time()
stp=stop_time
fr = RATE
fn=51200*N/50  #*RATE/44100
fs1=4.6439909297052155*N/50*11025/RATE
fs=fn/fr
print(fn,fs,fs1)
    
for s in range(0,900,1):
    start_measure()
    """
    fig.delaxes(ax1)
    fig.delaxes(ax3)
    ax1 = fig.add_subplot(311)
    ax1.set_title('passed time; {:.2f}(sec)'.format(time.time()-start))
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Signal')
    ax3 = fig.add_subplot(313)
    """
    input = []
    start_time=time.time()
    input = stream.read(CHUNK)
    stop_time=time.time()
    print(stop_time-start_time)
    #stp=stop_time
    
    sig = np.frombuffer(input, dtype="int16")  /32768.0
    #t = np.linspace(0,fs, fn, endpoint=False)
    """
    ax1.set_ylim(-0.0075,0.0075)
    ax1.set_xlim(0,fs)
    ax1.plot(t, sig)
    """
    nperseg = 1024
    f, t, Zxx = signal.stft(sig, fs=fn, nperseg=nperseg)
    ax2.pcolormesh(fs*t, f/fs/2, np.abs(Zxx), cmap='hsv')
    ax2.set_xlim(0,fs)
    ax2.set_ylim(20,20000)
    ax2.set_yscale('log')
    ax2.set_axis_off()
    """
    freq =fft(sig,int(fn))
    Pyy = np.sqrt(freq*freq.conj())*2/fn
    f = np.arange(20,20000,(20000-20)/int(fn)) #RATE11025,22050;N50,100
    ax3.set_ylim(0,0.000075)
    ax3.set_xlim(20,20000)
    ax3.set_xlabel('Freq[Hz]')
    ax3.set_ylabel('Power')
    ax3.set_xscale('log')
    ax3.plot(f*RATE/44100,Pyy)
    """
    
    plt.pause(0.01)
    plt.savefig('out_test/figure.jpg') #out_test/figure.jpg  #'train_images/0/figure' +str(s)+'.jpg'
    #output = stream.write(input)

stream.stop_stream()
stream.close()
p.terminate()

print( "Stop Streaming")