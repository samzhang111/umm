import pyaudio
import numpy as np
from matplotlib import pyplot as plt
import scipy
import pickle
from collections import deque

def adjust_units(rate, fft_data):
    return [i*(rate/2)/len(fft_data) for i in range(len(fft_data))]


def index_for_freq(units, freq):
    for i, u in enumerate(units):
        if u >= freq:
            return i-1

    return i

def is_um(X, clf):
    um_prob = clf.predict_proba(X)
    return um_prob


RATE=16000
RECORD_SECONDS = 15
CHUNKSIZE = RATE

with open('model.p', 'rb') as f:
    clf = pickle.load(f)

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)
frames = [] # A python-list of chunks(np.ndarray)
i = 0
for _ in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
    data = stream.read(CHUNKSIZE)
    np_data = np.fromstring(data, dtype=np.int16)
    fft_data = map(np.abs, scipy.fft(np_data))[:len(np_data)/2]
    adjusted_units = adjust_units(RATE, fft_data)

    prediction = is_um(fft_data, clf)
    print i, prediction

    i += 1


