import numpy as np
import hashlib
import time
import sys
import os
import wave
import struct


def load_data(filename):

    starting_size = os.path.getsize(filename)
    file = wave.open(filename,'r')
    file_arr = []

    n_frames = file.getnframes()

    """ Read the Wav File Data"""
    for i in range(0, n_frames):
        wavedata = file.readframes(1)
        data = struct.unpack("<h", wavedata)
        file_arr.append(int(data[0]))
        #print(int(data[0]))

    """ Generate A set of unique values"""
    uniques = set(file_arr)
    print(len(uniques))
    uniques_arr = []
    for i in uniques:
        uniques_arr.append(i)
    

    for i in range(10):
        print(file_arr[i])
    
    """ Replace Ints with Unique Index Values"""
    for i in range(0,n_frames):
        for j in range(0, len(uniques_arr)):
            if file_arr[i] == uniques_arr[j]:
                file_arr[i] = j

    bin_objects = ''
    hold_string = ''
    for i in range(len(file_arr)):
        #print('{0:8b}'.format(file_arr[i]))
        hold_string = str('{0:b}'.format(file_arr[i]))
        len_delta = 7-len(hold_string)
        for i in range(len_delta):
            hold_string = "0"+hold_string
        bin_objects = bin_objects + hold_string

    """ Add padding to make divisable by 8"""
    padding_needed = (8 - len(bin_objects) % 8) % 8
    for i in range(padding_needed):
        bin_objects = bin_objects +'0'

    """ Create padding string to append so the number of bits added is known """
    padding_string = str('{0:b}'.format(padding_needed))

    """ Add an extra five bits to the start of the padding to bring it up to one byte"""
    for i in range(5):
        padding_string = '0' + padding_string
    bin_objects = bin_objects + padding_string


    out_file_name = 'out.wav'
    if os.path.isfile(out_file_name):
        os.remove(out_file_name)
    for i in range(0,len(bin_objects), 8):
        open(out_file_name, 'ba+').write(int(bin_objects[i:i+8], 2).to_bytes(len(bin_objects[i:i+8]) // 8, byteorder='big'))

    

    print('N Channels: ' + str(file.getnchannels()))
    print('N Frames: ' + str(file.getnframes()))
    print('Frame Rate: ' + str(file.getframerate()))
    print('N Frames: ' + str(file.getnframes()))
    
def decode(out_file_name):
    print('decode')
    #encoded = open(out_file_name, 'r').read()

def validate_comp(out_file_name, filename):
    print('validate')



    



if __name__== "__main__":
    start_time = time.time()
    filename = '0ab237b7-fb12-4687-afed-8d1e2070d621.wav'
    out_file_name = load_data(filename)

    decode_file = decode(out_file_name)

    validate = validate_comp(out_file_name, filename)

    #data = load_data('0aefe960-43fd-41cc-97c8-bf9d2d64efd3.wav')
    #data = load_data('0b049a37-dc68-42f6-bbe5-d7d1cd699ccb.wav')
    #data = load_data('0b4adf65-d666-4fa4-971f-6e7ed630d7fb.wav')
    #data = load_data('0befa6a5-0c9b-44c6-93b8-eb6678045480.wav')
    #data = load_data('0c3f4ca2-459d-4643-9b7b-a2c7ec378172.wav')
    #data = load_data('0cc70045-605a-429f-90f3-a78b12f071a3.wav')
    print("--- %s seconds ---" % (time.time() - start_time))