# Script to trasform the data into spaCy's required format for CLI
# Split the corpus dataset into train and test set.
from scripts import *
import srsly

# User input path or use default.
clean_spacy_data = "processed_data/ade_spacy_data.obj"
data_path = input("Enter the processed data path: ") or clean_spacy_data

# Get ouput directory
output_dir = "processed_data/spaCy_CLI/"
output_path = input("Enter the output path: ") or output_dir

# Get filename
current_time = get_time()
file_name = input("Enter the output file name: ") or current_time
file_name = file_name + ".jsonl"

# Load data to path
DATA = load_pkl(data_path)
nlp = spacy.load('en_core_web_md')

# Shuffle the dataset and partition it into 10-fold for cross-validation.
random.shuffle(DATA)
data_folds = partition_data(DATA)
train_data, test_data = split_data(data_folds, 0)

# Convert ade corpus data to spaCy's CLI training format.
print("Converting dataset into spaCy's required format, printing error ...")
train_data,_,train_errors = convertspacyapiformattocliformat(nlp, train_data)
test_data,_,test_errors = convertspacyapiformattocliformat(nlp, test_data)
print("Total number of error during data conversion %d"%(len(test_errors)+len(train_errors)))

srsly.write_jsonl(output_dir+file_name, train_data)
srsly.write_jsonl(output_dir+"test_"+file_name, test_data)