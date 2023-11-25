# Dependencies for this notebook
import numpy as np
import pandas as pd

# Ignore warning for this project
import warnings
warnings.filterwarnings("ignore")

import os
import joblib

# Load the dataset
df = pd.read_csv("../data/Verificatie Dataset dieten met density.csv", sep=";")

validation = df["Prediction.OUT"]
input = df[["Density", "PV", "Line 1", "Line 2", "Line 3"]]

models = os.listdir("out/dieten met density")
scaler = joblib.load("out/Scaler.pkl")
scaled_input = scaler.transform(input)

if (os.path.exists("out/predictions") == False):
	os.mkdir("out/predictions")

if (os.path.exists("out/dieten met density") == False):
	os.mkdir("out/dieten met density")

for model_file in models:
	model_name = model_file.replace(".pkl", "")
	model = joblib.load(f"out/dieten met density/{model_file}")
	predictions = model.predict(scaled_input)

	output = pd.DataFrame({ 
		#"qm_spec_id":  df["qm_spec_id"],
		"Line 1":  df["Line 1"], 
		"Line 2":  df["Line 2"], 
		"Line 3":  df["Line 3"], 
		"Density":  df["Density"],
		"PV": df["PV"],
		"Prediction.OUT": predictions, 
		"Validation.OUT": validation 
	})

	output.to_csv(f"../out/dieten met density/{model_name}.csv", index=False, sep=";", decimal=",")