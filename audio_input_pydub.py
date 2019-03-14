import os, sys
import subprocess

input_file='output.avi'
output_file='output1.avi'
bitrate=40
channel=1

cmd = "ffmpeg -y -i {} -ab {}k -ac {} {}".format(input_file, bitrate, channel, output_file)
resp = subprocess.check_output(cmd, shell=True)

import pydub
from pydub import AudioSegment
wavfile='karasu-miyama_out.wav'
base_sound = AudioSegment.from_file(wavfile, format="wav")  # 音声を読み込み input.mp3, format="mp3"
length_seconds = base_sound.duration_seconds  # 長さを確認
base_sound.export("karasu-miyama_out.mp3", format="mp3")  # 保存する

from pydub.utils import db_to_float, ratio_to_db
#ratio = 0.8  # 0.8倍の音量にしたい
#quiet_sound = base_sound + ratio_to_db(ratio)

delta = ratio_to_db(50)  # 音量を0.8倍にしたい
sound_loud = base_sound + delta  # 音量を調節 sound_base + delta
result_ratio = sound_loud.rms / base_sound.rms  #sound_base.rms
print(result_ratio)  # 0.7998836532867947が返ってきた
sound_loud.export("karasu-miyama_out1.mp3", format="mp3")  # 保存する

import moviepy.editor as mp
clip_output = mp.VideoFileClip(output_file).subclip()
clip_output.write_videofile(output_file.replace('.avi', '.mp4'), audio='karasu-miyama_out1.mp3')
