import wave
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
from collections import deque
import pandas as pd

def main():
    all_data = []
    for fn in sys.argv[1:]:
        print fn
        wv = wave.open(fn, 'rb')
        rows = extract_fft_rows(wv)

        label = fn.startswith('um_training/um')
        for i, r in enumerate(rows):
            row = np.append(r, label)
            all_data.append(row)

    df = pd.DataFrame(all_data)
    df.ix[:, df.shape[1]-1].fillna(method='pad', inplace=True)
    df.to_csv('um_training.csv', index=False)


def extract_fft_rows(wv):
    RATE = wv.getframerate()
    # Using a single second windows for training data
    CHUNKSIZE = 200

    print 'Reading wave with sampling frequency: ', RATE

    window = deque()
    frames = [] # A python-list of chunks(np.ndarray)
    # Skip first row
    data = wv.readframes(RATE)
    data = wv.readframes(RATE)
    i = 0
    while data:
        np_data = np.fromstring(data, dtype=np.int16)
        window.extend(np_data)
        for _ in range(len(window) - RATE):
            window.popleft()

        fft_data = map(np.abs, scipy.fft(window)[:len(window)/2])
        data = wv.readframes(CHUNKSIZE)
        if not data:
            # Skip the last record as well
            return frames

        frames.append(fft_data)
        i += 1

    return frames


if __name__ == '__main__':
    main()
