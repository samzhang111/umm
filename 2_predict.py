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
CHUNKSIZE = 500

with open('model.p', 'rb') as f:
    clf = pickle.load(f)

window = deque()

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

i = 0
data = stream.read(RATE)
np_data = np.fromstring(data, dtype=np.int16)
window.extend(map(np.abs, np_data))

print len(window)
for _ in range(0, RATE * RECORD_SECONDS / CHUNKSIZE):
    d = stream.read(CHUNKSIZE)
    window.extend(map(np.abs, np.fromstring(d, dtype=np.int16)))
    for _ in range(CHUNKSIZE):
        window.popleft()

    fft_data = scipy.fft(list(window))[:len(window)/2]
    prediction = is_um(fft_data, clf)

    if prediction[0][1] > 0.1:
        print "Um...", prediction

    i += 1

