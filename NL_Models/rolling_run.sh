#!/bin/bash
while :
do 
    python3 src_v2/DRGNFLY.py -c NL_Models/Model1/config_m1.yaml -p NL_Models/Model1/Input-8-NodePHidden-10-NHidden-2-NOut-1.csv -d NL_Models/training_data_full_bands.csv -t -e
done