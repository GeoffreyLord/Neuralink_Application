# DRGNFLY
Simulation Problem #1

![my-pixel-img (1)](https://user-images.githubusercontent.com/48192445/161814543-ada38238-63c0-47d4-844e-8a878a0bc5c0.png)

**Overview:**
DRGNFLY  - Neural Network Sandbox


**Notes:**

src uses micrograd and src_v2 uses pytorch. Depending on the system being used the pytorch version (src_v2) may be faster.


**Usage:**

```python3 ../../../src/DRGNFLY.py -c config.yaml -p Input-x...-NOut-x.csv -d datafile.csv -t -e```

```-c: Configuration File with model attributest```

```-p: Model Paramaters File (If not provided application will generate after training a model)```

```-d: Training/Testing Data File (If not provided data will be created from config.yaml file)```

```-t: Training Mode (Will Train Model)```

```-e: Evaluate Mode (Will Test/Evaluate Model)```


File List v1
* src/DRGNFLY.py: 
* src/management/dataMgmt.p
* src/management/drgnflyObj.py
* src/management/initalize.py
* src/management/model.py
* src/management/viz.py
* src/micrograd/engine.py
* src/micrograd/nn.py
* src/testing/profileModel.py
* src/testing/testModel.py

