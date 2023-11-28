# Dependencies for this notebook
import argparse
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import io, os

# Ignore warning for this project
import warnings
warnings.filterwarnings("ignore")

# Exploration method will generate information about the dataset
def explore(input: str, output: str, prefix = ""):
	
	# Load the dataset
	df = pd.read_csv(input, sep=";", decimal=",")

	# Generate information and write to output directory
	generate_text_info(df, output)
	generate_correlation_diagram(df, output, prefix)
	generate_box_plot_diagrams(df, output, prefix)

# Generates text info about the dataset
def generate_text_info(df: pd.DataFrame, output: str):
	# Print information about the dataset like column names, amount of rows and datatypes
	buffer = io.StringIO()
	df.info(buf=buffer)
	df.describe().to_string(buffer)
	
	s = buffer.getvalue()

	if os.path.exists(output) == False:
		os.makedirs(output)

	with open(f"{output}/Info.txt", "w",encoding="utf-8") as f:
		f.write(s)

# Generates a box plot diagram for every column in the dataset
def generate_box_plot_diagrams(df: pd.DataFrame, output: str, prefix: str = ""):
	for column in df.columns:
		fig = plt.figure()
		boxplot = sns.boxplot(df[column]);
		boxplot.set_title(f"{prefix}Boxplot {column}")
		plt.savefig(f"{output}/{prefix}Boxplot_{column}.png")

# Generates a correlation diagram between the available column
def generate_correlation_diagram(df: pd.DataFrame, output: str, prefix: str = ""):
	# Create a correlation plot and save it as PNG
    corr = df.corr()
    fig = plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    ax = fig.add_subplot()
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(df.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.columns)

    plt.title(f"{prefix}Correlation")  # Set the title here

    # Write values inside diagram
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            ax.text(j, i, round(corr.iloc[i, j], 2),
                    ha='center', va='center', color='black')

    # Save to disk
    plt.savefig(f"{output}/{prefix}Correlation.png", bbox_inches='tight')  # Add bbox_inches to ensure everything is saved

# Parses command line input
def parse_arguments():
	argParser = argparse.ArgumentParser()
	argParser.add_argument("-i", "--input", required=True, type=str, help="Input CSV file")
	argParser.add_argument("-o", "--output", default=".", type=str, help="Output directory")
	return argParser.parse_args()

# Will be run when script is called from command line
if __name__ == "__main__":
	args = parse_arguments()
	explore(args.input, args.output)