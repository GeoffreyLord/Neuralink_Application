# DRGNFLY: Neural Network Interpretability Engine

#Example:
```python3 ../../../src_v2/DRGNFLY.py -c config.yaml -p Input-x...-NOut-x.csv -d datafile.csv -t -e```

**Overview:**
This repo documents the application of a neural network analysis engine as it relates to the mapping of brain wave data to intended user actions. Utilizing a modified children's toy, I collected EEG sensor data, trained neural networks to decode neural activity, and optimized brain wave models using a custom neural network interpretability tool. The optimized models exhibited similar accuracy to the larger parent model while allowing execution time to be decreased by up to 95.18\%. My hope is that this repo and its corresponding writeup articulate my interest in joining the Neuralink team whilst providing insights that may lead to improvement of the N1 implant's bandwidth.


**Notes:**
src uses micrograd and src_v2 uses pytorch. Depending on the system being used the pytorch version (src_v2) may be faster.

**Usage:**
```python3 ../../../src_v2/DRGNFLY.py -c config.yaml -p Input-x...-NOut-x.csv -d datafile.csv -t -e```

```-c: Configuration File with model attributes``

```-p: Model Parameters File (If not provided application will generate after training a model)```

```-d: Training/Testing Data File (If not provided data will be created from config.yaml file)```

```-t: Training Mode (Will Train Model)```

```-e: Evaluate Mode (Will Test/Evaluate Model)```







