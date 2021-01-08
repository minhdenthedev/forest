import numpy as np
import pyaudio
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import os

MODEL_NAME = 'mobilenet'
GRAPH_NAME = 'mobilenet.tflite'
LABELMAP_NAME = 'labelmap.txt'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)
IMG_SIZE = (224, 224)
INPUT_MEAN = 127.5
INPUT_STD = 127.5
count = 0

class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = self.RATE * 2
        self.p = None
        self.stream = None

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        global interpreter, input_data, output_details, labels, count
        N_FFT = 1024         # 
        HOP_SIZE = 1024      #  
        N_MELS = 128          # Higher   
        WIN_SIZE = 1024      # 
        WINDOW_TYPE = 'hann' # 
        FEATURE = 'mel'      # 
        FMIN = 1400
        
        numpy_array = np.frombuffer(in_data, dtype=np.float32)
        # numpy_array = nr.reduce_noise(audio_clip=numpy_array, noise_clip=numpy_array, prop_decrease = 1, verbose=False)
        S = librosa.feature.melspectrogram(y=numpy_array, sr=self.RATE,
                                        n_fft=N_FFT,
                                        hop_length=HOP_SIZE, 
                                        n_mels=N_MELS, 
                                        htk=True, 
                                        fmin=FMIN, # higher limit ##high-pass filter freq.
                                        fmax=self.RATE / 4) # AMPLITUDE
                                        
        fig = plt.figure(1,frameon=False)
        fig.set_size_inches(2.24,2.24)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        librosa.display.specshow(librosa.power_to_db(S**2,ref=np.max), fmin=FMIN) #power = S**2
        file_name = 'data/1Bi/Image_{0}.png'.format(count)
        fig.savefig(file_name)
        count += 1
                            
        
        image = cv2.imread(file_name)
        cv2.imshow('Melspec', image)
        cv2.waitKey(1)

        return None, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active()): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition
            time.sleep(2.0)


audio = AudioHandler()
audio.start()
audio.mainloop()
audio.stop()
