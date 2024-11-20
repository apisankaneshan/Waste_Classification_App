# **Waste Classification App**

This repository consits of an application which performs CNN transfer learning using the MobileNetV2 architecture as the functional base model.
The goal was to retrain the base model which is used for image classification tasks, to specialize in waste classification. Specifically, recyclable waste and organic waste.
This goal was successfully acheieved. Using the newly trained model, a python application was built for computers with cameras to detect and classify waste.


## Installation
_Ensure the python version and python interperter version you are using is python 3.10.15._
_This installation is done on macOS._
```python
# Clone repo
git clone https://github.com/apisankaneshan/Waste_Classification_App.git

# Enter Project folder
cd Project

# Set up virtual environment
python3 -m venv <virtual_environment_directory>

# Enter virtual environment
source <virtual_environment_directory>/bin/activate

# Install all needed packages
pip install opencv-python
pip install numpy
pip install pandas
pip install matplotlib
pip install tqdm
pip install tensorflow
pip install keras
pip install pydot
pip install scipy
```

## Running the App
```python
python model_test1.py
```

## Creating your own model
To create your own model, edit the ```waste_classification_model_creator.py``` file. 
Simply correct the paths for your dataset then run the file. 
The resulting model will be stored in the file ```waste_classification.keras```

