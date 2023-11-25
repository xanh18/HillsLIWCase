import os;
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

# Load the dataset
data = pd.read_csv('../data/dataset met dieten.csv', sep=";", decimal=",")

# Feature selection
print("Extracting features")
features = data[[
	'Density',
	'PV',
	'Line 1',
	'Line 2',
	'Line 3'
]]

# Target variable
target = data['OUT']

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
	"SVR": SVR(),
	"Lasso": Lasso(),
	"RandomForest": RandomForestRegressor(),
	"HistGradientBoosting": HistGradientBoostingRegressor(),
}

if (os.path.exists("out") == False):
	os.mkdir("out")


if (os.path.exists("out/models") == False):
	os.mkdir("out/models")

if (os.path.exists("out/dieten met density") == False):
	os.mkdir("out/dieten met density")

joblib.dump(scaler, 'out/Scaler.pkl')

for model_name in models:
	model = models[model_name]
	model.fit(X_train, y_train)
	predictions = model.predict(X_test)
	score = model.score(X_test, y_test)

	# Evaluate the model
	mse = mean_squared_error(y_test, predictions)
	mae = mean_absolute_error(y_test, predictions)
	r2 = r2_score(y_test, predictions)

	table = pd.DataFrame([[score, mse, mae, r2]], index=[model_name], columns=["Score", "MSE", "MAE", "R2"])
	
	print("")
	print(table)

	joblib.dump(model, f'out/dieten met density/{model_name}.pkl')
