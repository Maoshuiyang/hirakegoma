# hirakegoma
Making a 'hirakegoma' service by VGG-like cnn, Pyaudio, STFT, keras based data-processing thechnique, and so on

1. VGG16_originalData.py for training, use 'getDataSet.py'.
2. Use 'pyaudio_realtime_last.py' in backgraound, and start 'out_onsei.py'.
3. 'out_onsei.py' plays 'hirakegoma.wav' or 'ohayo.wav' according to the file 'figure.jpg' in dir './out_test/'.
4. 'figure.jpg' is a stft image made by real time stft by 'pyaudio_realtime_last.py'.
5. This application uses a microphon.
