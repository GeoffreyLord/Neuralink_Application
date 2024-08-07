#!/bin/bash
while :
do 
    python3 ../../../src_v2/DRGNFLY.py -c config_wave.yaml -p Input-8-NodePHidden-15-NHidden-2-NOut-1.csv -d normalized_clickeyes_data_wave.csv -t
done