import os, joblib, argparse, pandas as pd, json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from parser import keyvalue

# Ignore warning for this project
import warnings
warnings.filterwarnings("ignore")

def train(input: str, output: str, target_name: str = 'OUT', feature_names: list[str] = [], model_names: list[str] = [], mapper: dict[str, str] = {}):

	if os.path.exists(input) == False:
		print(f"Input '{input}' does not exist!")
		return

	if os.path.isdir(input):
		print(f"Input '{input}' must be a file")
		return
	
	print(f"Training: {input}")

	# Load the dataset
	data = pd.read_csv(input, sep=";", decimal=",")
	data = data.rename(columns=mapper)

	#check Datetime and encode this.

	if 'DateTime' in data.columns:
		print('ok')

	# Check if target is in dataset
	if (target_name in data.columns.values) == False:
		print(f"Dataset does not contain target '{target_name}'")
		return

	# Get target values from dataset
	target = data[target_name]

	# Feature selection
	if len(feature_names) == 0:
		feature_names = list(data.columns.values)
	
	feature_names.remove(target_name)

	print(f"Training with features: {feature_names}")

	# Get features from dataset
	features = data[feature_names]

	# Data preprocessing
	scaler = StandardScaler()
	features_scaled = scaler.fit_transform(features)

	# Split the data
	X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

	# Models
	models = {
		"LinearRegression": LinearRegression(),
		"DecisionTree": DecisionTreeRegressor(),
		"GradientBoosting": GradientBoostingRegressor(),
		"LinearSVR": LinearSVR(max_iter=1000),
		"Lasso": Lasso(),
		"RandomForest": RandomForestRegressor(),
		"HistGradientBoosting": HistGradientBoostingRegressor(),
	}

	if(len(model_names) == 0):
		model_names = models.keys()

	if (os.path.exists(output) == False):
		os.makedirs(output)

	train_info = {
		"features": feature_names,
		"models": list()
	}

	for model_name in model_names:
		
		# Check if model is in models dict
		if((model_name in models.keys()) == False):
			continue

		print(f"Training model {model_name}")

		model = models[model_name]
		model.fit(X_train, y_train)
		predictions = model.predict(X_test)

		# Evaluate the model
		mse = mean_squared_error(y_test, predictions)
		mae = mean_absolute_error(y_test, predictions)
		r2 = r2_score(y_test, predictions)

		table = pd.DataFrame([[mse, mae, r2]], index=[model_name], columns=["MSE", "MAE", "R2"])
	
		print("Score: ")
		print(table)

		train_info["models"].append({
			"name": model_name,
			"mse": mse,
			"mae": mae,
			"r2": r2
		})

		model.feature_names = list(feature_names)
		joblib.dump(scaler, f"{output}/{model_name}Scaler.pkl")
		joblib.dump(model, f'{output}/{model_name}.pkl')
		print("Model written to disk")
	
	# Writing to sample.json
	with open(f"{output}/train.json", "w") as outfile:
		outfile.write(json.dumps(train_info, indent=4))

def parse_arguments():
	argParser = argparse.ArgumentParser()
	argParser.add_argument("-i", "--input", required=True, type=str, help="Input CSV file")
	argParser.add_argument("-o", "--output", default=".", type=str, help="Output directory")
	argParser.add_argument("-t", "--target", type=str, default='OUT', help="Target value")
	argParser.add_argument("-f", "--features", type=str, nargs='+', default=[], help="Features to use")
	argParser.add_argument("-m", "--models", type=str, nargs='+', default=[], help="Model names")
	argParser.add_argument('--mapper', type=str, default={}, nargs='*', action = keyvalue) 
	return argParser.parse_args()

if __name__ == "__main__":
	print("Train")
	args = parse_arguments()
	train(args.input, args.output, args.target, args.features, args.models, args.mapper)