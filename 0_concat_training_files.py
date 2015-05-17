import wave
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import pandas as pd

def adjust_units(rate, fft_data):
    return [i*(rate/2)/len(fft_data) for i in range(len(fft_data))]


def index_for_freq(units, freq):
    for i, u in enumerate(units):
        if u >= freq:
            return i-1

    return i

RECORD_SECONDS = 15
EXPECTED_HIGH = 378
EXPECTED_LOW = 143
EXPECTED_RATIO = float(EXPECTED_HIGH)/EXPECTED_LOW

MID_CUTOFF = 200.0
MIN_AMP = 100

def main():
    all_data = []
    for fn in sys.argv[1:]:
        print fn
        wv = wave.open(fn, 'rb')
        rows = extract_fft_rows(wv)

        for i, r in enumerate(rows):
            rows[i] = np.append(r, fn.startswith('um_training/um'))

        all_data.extend(rows)

    df = pd.DataFrame(all_data)
    df.to_csv('um_training.csv', index=False)

def extract_fft_rows(wv):
    RATE = wv.getframerate()
    # Using a single second windows for training data
    CHUNKSIZE = RATE

    print 'Reading wave with sampling frequency: ', RATE

    frames = [] # A python-list of chunks(np.ndarray)
    # Skip first row
    data = wv.readframes(CHUNKSIZE)
    data = wv.readframes(CHUNKSIZE)
    i = 0
    while data:
        np_data = np.fromstring(data, dtype=np.int16)
        fft_data = map(np.abs, scipy.fft(np_data))[:len(np_data)/2]
        data = wv.readframes(CHUNKSIZE)
        if not data:
            # Skip the last record as well
            continue

        adjusted_units = adjust_units(RATE, fft_data)

        print i, len(fft_data)

        # Calculate additional diagnostics
        frames.append(fft_data)
        i += 1

    return frames

if __name__ == '__main__':
    main()
