import os
import wave
import time
import struct
import hashlib
import numpy as np
import matplotlib.pyplot as plt



def load_data(filename):
    file = wave.open(filename, 'r')
    try:
        print(f'N Channels: {file.getnchannels()}')
        print(f'N Frames: {file.getnframes()}')
        print(f'Frame Rate: {file.getframerate()}')
        n_frames = 1000#file.getnframes()

        file_arr = []
        """ Read the Wav File Data"""
        for i in range(0, n_frames):
            wavedata = file.readframes(1)
            data = struct.unpack("<h", wavedata)
            file_arr.append(int(data[0]))

        index_array = [i for i in range(1, n_frames+1)]

    finally:
        file.close()  # Ensure the file is closed properly

    return index_array, file_arr


def plot_data(filenames):
    plt.figure()
    for filename in filenames:
        index_array, file_arr = load_data(filename)
        plt.plot(index_array, file_arr, label=os.path.basename(filename))
    
    plt.title("Electrode Data Amplitude vs Sample")
    plt.xlabel("X - Sample")
    plt.ylabel("Y - Amplitude")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    filenames = [
        '0ab237b7-fb12-4687-afed-8d1e2070d621.wav',
        '0b4adf65-d666-4fa4-971f-6e7ed630d7fb.wav',
        '0aefe960-43fd-41cc-97c8-bf9d2d64efd3.wav'
    ]
    plot_data(filenames)

"""
        '0befa6a5-0c9b-44c6-93b8-eb6678045480.wav',
        '0b049a37-dc68-42f6-bbe5-d7d1cd699ccb.wav',
        '0c3f4ca2-459d-4643-9b7b-a2c7ec378172.wav',
        '0cc70045-605a-429f-90f3-a78b12f071a3.wav'
        
"""