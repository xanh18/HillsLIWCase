# Ignore warning for this project
import os, joblib, argparse, pandas as pd
import warnings

from scripts.parser import keyvalue

warnings.filterwarnings("ignore")

def predict(input: str, output: str, model_names: list[str] = [], mapper: dict[str, str] = {}):

	# Load the dataset
	data = pd.read_csv(input, sep=";", decimal=",")
	data = data.rename(columns=mapper)
	
	print(f"Predict: {input}")

	if (os.path.exists(output) == False):
		os.makedirs(output)

	models = {}
	scalers = {}
	for model_path in os.listdir(output):
		base_name = os.path.basename(model_path)
		
		# Ignore directories
		if(os.path.isdir(base_name)):
			continue

		model_name = base_name.replace(".pkl", "")
		
		# Ignore non model files
		if base_name.endswith(".pkl") == False or base_name.endswith("Scaler.pkl"):
			continue

		scaler = os.path.join(output, f"{model_name}Scaler.pkl")
		if os.path.exists(scaler) == False:
			print(f"Scaler is missing for model {model_name}")
			continue

		# Append to models dictionary
		if len(model_names) == 0 or model_name in model_names:
			models[model_name] = os.path.join(output, base_name)
			scalers[model_name] = scaler
	
	for model_name in models:
		model = joblib.load(models[model_name])
		scaler = joblib.load(scalers[model_name])
		
		# Get features from dataset
		features = data[model.feature_names]
		scaled_input = scaler.transform(features)
		data[f"{model_name}"] = model.predict(scaled_input)

	data.to_csv(f"{output}/Predictions.csv", index=False, sep=";", decimal=",")

def parse_arguments():
	argParser = argparse.ArgumentParser()
	argParser.add_argument("-i", "--input", required=True, type=str, help="Input CSV file")
	argParser.add_argument("-o", "--output", default=".", type=str, help="Output directory")
	argParser.add_argument("-m", "--models", type=str, nargs='+', default=[], help="Model names")
	argParser.add_argument('--mapper', type=str, default={}, nargs='*', action = keyvalue) 
	return argParser.parse_args()

if __name__ == "__main__":
	args = parse_arguments()
	predict(args.input, args.output, args.models, args.mapper)