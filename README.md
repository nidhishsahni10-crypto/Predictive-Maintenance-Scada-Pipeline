For my internship at TensorMaxwell Corporation I built a Python pipeline that works with industrial SCADA data.
SCADA stands for Supervisory Control and Data Acquisition, basically it is the system factories use to collect
sensor readings from machines in real time. The goal of this project was to take that raw sensor data, clean it
up, detect if anything has gone wrong with the sensors over time, train some machine learning models on it, and
finally use optimisation to reduce the energy cost of running the plant. The whole thing is split across four
Python files, each handling a different part of the pipeline.


This is a code implementation on basic level of my understanding for my work observational work at TensorMaxwell . 

Following are files 

generate_data.py  -  Generates simulated SCADA and energy-yield loops

pipeline_drift.py  - Cleans data, handles missing values and detects drift

models_bench.py - PyTorch LSTM + 5 Scikit-learn Classifiers grid search

optimization.py - Scipy optimization minimizing energy costs by 15-20%


<img width="1205" height="575" alt="image" src="https://github.com/user-attachments/assets/0da3aeda-ae45-4a3d-bc10-721612384728" />
