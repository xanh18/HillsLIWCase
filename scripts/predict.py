# Dependencies for this notebook
import numpy as np
import pandas as pd

# Ignore warning for this project
import warnings
warnings.filterwarnings("ignore")

import os
import joblib

# Load the dataset
df = pd.read_csv("data/input.csv", sep=";")

validation = df["Dry_Feed_Rate_PID.OUT"]
input = df[["Density_Target", "Dry_Feed_Rate_PID.SP", "Line 1", "Line 2","Line 3"]]

models = os.listdir("out/models")
scaler = joblib.load("out/Scaler.pkl")
scaled_input = scaler.transform(input)

if (os.path.exists("out/predictions") == False):
	os.mkdir("out/predictions")

for model_file in models:
	model_name = model_file.replace(".pkl", "")
	model = joblib.load(f"out/models/{model_file}")
	predictions = model.predict(scaled_input)

	output = pd.DataFrame({ 
		"qm_spec_id":  df["qm_spec_id"], 
		"Line 1":  df["Line 1"], 
		"Line 2":  df["Line 2"], 
		"Line 3":  df["Line 3"], 
		"Density_Target":  df["Density_Target"], 
		"Dry_Feed_Rate_PID.SP": df["Dry_Feed_Rate_PID.SP"],
		"Prediction.OUT": predictions, 
		"Validation.OUT": validation 
	})

	output.to_csv(f"out/predictions/{model_name}.csv", index=False, sep=";", decimal=",")