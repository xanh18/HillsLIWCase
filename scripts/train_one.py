# Import the train method from train.py
import argparse
from explore import explore
from train import train
from predict import predict

# This method will train all known datasets
def train_one(input: str, validation: str, output: str, models: list[str]):
	explore(input=input,output=output)
	train(input=input, output=output,model_names=models)
	predict(input=validation, mapper={ "SP": "PV" },output=output,model_names=models)

def parse_arguments():
	argParser = argparse.ArgumentParser()
	argParser.add_argument("-i", "--input", required=True, type=str, help="Input CSV file")
	argParser.add_argument("-v", "--validation", required=True, type=str, help="Input validation CSV file")
	argParser.add_argument("-o", "--output", default=".", type=str, help="Output directory")
	argParser.add_argument("-m", "--models", type=str, nargs='+', default=[], help="Model names")
	return argParser.parse_args()

# This method will be called when python executes this script
if __name__ == "__main__":
	args = parse_arguments()
	train_one(args.input, args.validation, args.output, args.models)