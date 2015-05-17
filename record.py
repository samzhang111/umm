import pyaudio
import numpy as np
from matplotlib import pyplot as plt
import scipy

def adjust_units(rate, fft_data):
    return [i*(rate/2)/len(fft_data) for i in range(len(fft_data))]


def index_for_freq(units, freq):
    for i, u in enumerate(units):
        if u >= freq:
            return i-1

    return i

RATE=2000
RECORD_SECONDS = 15
CHUNKSIZE = 2000

EXPECTED_HIGH = 378
EXPECTED_LOW = 143
EXPECTED_RATIO = float(EXPECTED_HIGH)/EXPECTED_LOW

MID_CUTOFF = 200.0
MIN_AMP = 100

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

frames = [] # A python-list of chunks(np.ndarray)
for _ in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
    data = stream.read(CHUNKSIZE)
    np_data = np.fromstring(data, dtype=np.int16)
    fft_data = map(np.abs, scipy.fft(np_data))[:len(np_data)/2]
    adjusted_units = adjust_units(RATE, fft_data)

    unit_of_freq = len(fft_data)/(RATE/2.0)

    # TODO: High pass filter
    low_cutoff = index_for_freq(adjusted_units, 100)
    fft_data = fft_data[low_cutoff:]
    adjusted_units = adjusted_units[low_cutoff:]

    # Split data into two windows for the formant
    mid_cutoff = index_for_freq(adjusted_units, MID_CUTOFF)
    # Get the peaks for two windows
    fft_low = fft_data[:mid_cutoff]
    adjusted_units_low = adjusted_units[:mid_cutoff]

    low_index = np.argmax(fft_low)
    max_low = adjusted_units_low[low_index]
    max_amp_low = fft_low[low_index]

    fft_high = fft_data[mid_cutoff:]
    adjusted_units_high = adjusted_units[mid_cutoff:]

    high_index = np.argmax(fft_high)
    max_high = adjusted_units_high[high_index]
    max_amp_high = fft_high[high_index]

    # Get additional diagnostics (mean amplitude)
    mean_amp = np.mean(fft_data)

    # Get the ratio of the two peaks
    ratio = float(max_high)/max_low
    high_low_amp_ratio = float(max_amp_high)/max_amp_low
    high_all_amp_ratio = float(max_amp_high)/mean_amp

    print "Ratios:", max_low, max_high, ratio, high_low_amp_ratio, high_all_amp_ratio

    if np.abs(ratio - 2.0) < 0.1 and mean_amp > MIN_AMP:
        print "UM!!!!!!"

    # Calculate additional diagnostics
    frames.append(np_data)

#Convert the list of np-arrays into a 1D array (column-wise)
npdata = np.hstack(frames)
fft_all = map(np.abs, scipy.fft(npdata))[:len(npdata)/2]
all_units = adjust_units(RATE, fft_all)
plt.plot(all_units, fft_all)
plt.show()

# close stream
stream.stop_stream()
stream.close()
p.terminate()
