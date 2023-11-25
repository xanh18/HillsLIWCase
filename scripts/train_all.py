# Import the train method from train.py
from train import train
from predict import predict

# This method will train all known datasets
def train_all():

	# These are all variants to train
	variants = [
		"Dieten met density",
		"Dieten zonder density",
		"Line 1 met density",
		"Line 2 met density",
		"Line 3 met density",
		"Line 1 zonder density",
		"Line 2 zonder density",
		#"Dataset per dieet en lijn test"
	]

	# Train variants
	for variant in variants:
		train(
			input=f"data/20231120 Dataset {variant}.csv", 
			output=f"out/{variant}",
			model_names=[]
		)
		predict(
			input=f"data/Verificatie Dataset {variant}.csv", 
			mapper= { "SP": "PV" },
			output=f"out/{variant}",
			model_names=[],
		)

# This method will be called when python executes this script
if __name__ == "__main__":
	train_all()