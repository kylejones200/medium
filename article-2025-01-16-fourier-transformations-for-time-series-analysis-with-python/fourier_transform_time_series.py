import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from sktime.datasets import load_airline

# Example 1: Synthetic time series with two frequencies
np.random.seed(42)
time = np.linspace(0, 10, 500)
freq1, freq2 = 2, 5
signal = np.sin(2 * np.pi * freq1 * time) + 0.5 * np.sin(2 * np.pi * freq2 * time)

plt.figure(figsize=(10, 4))
plt.plot(time, signal)
plt.title("Time Series (Combination of Two Sine Waves)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# FFT on synthetic signal
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(fft_result), d=(time[1] - time[0]))

plt.figure(figsize=(10, 4))
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result)[:len(fft_result)//2])
plt.title("Frequency Domain Representation")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()

# Example 2: Airline passengers dataset
y = load_airline()
y = y.values
time = np.arange(len(y))
fft_result = np.fft.fft(y)
frequencies = np.fft.fftfreq(len(fft_result), d=1)

plt.figure(figsize=(10, 4))
plt.plot(y)
plt.title("Original Airline Passenger Data")
plt.xlabel("Time")
plt.ylabel("Passengers")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result)[:len(fft_result)//2])
plt.title("Frequency Domain of Airline Passenger Data")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.savefig("time_series_passanger.png")
plt.show()

# Example 3: Noise filtering
noisy_signal = y + np.random.normal(0, 50, len(y))
fft_result_noisy = np.fft.fft(noisy_signal)
fft_filtered = fft_result_noisy.copy()
threshold = 0.1
fft_filtered[np.abs(frequencies) > threshold] = 0
filtered_signal = np.fft.ifft(fft_filtered)

plt.figure(figsize=(10, 6))
plt.plot(time, noisy_signal, label="Noisy Signal", alpha=0.5)
plt.plot(time, filtered_signal.real, label="Filtered Signal", color='red')
plt.title("Noise Filtering with Fourier Transform")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.savefig("time_series_passanger_amp.png")
plt.show()

# Example 4: FFT with SciPy
fft_result_scipy = fft(signal)
ifft_result_scipy = ifft(fft_result_scipy)
