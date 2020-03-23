from model import *
from scripts import *

# Load pickle dataset from user input or default.
data_path = input("Enter the pickle training data path: ") or "./processed_data/ade_spacy_data.obj"
DATA = load_pkl(data_path)

# Preset entities label
entities = ["SYMP","DRUG","DOSE"]

# Update or train new blank model.
model = None

# Output Directory to save model after training
current_time = get_time()
user_dir = input("Enter the output path: ") or current_time
output_dir = "./pretrained_model/spacy/" + user_dir
print(output_dir)

# Number of iteration epoch to train the model, default 1.
n_iter = input("Enter the number of training iterations: ") or 10

# User select what mode to run the model as.
MODE = input("Enter training mode; 1.Default 2.Cross-validation 3.Full-model: ") or 1
MODE = int(MODE)

# Split data into 10-fold for cross-validation of model.
random.shuffle(DATA)
data_folds = partition_data(DATA)

# Split data to train and test set
train_data, test_data = split_data(data_folds, len(data_folds)-1)

print("Total number of data: %d"%(len(DATA)))
print("Number of train data: %d"%(len(train_data)))
print("Number of test data: %d"%(len(test_data)))
print("Number of tags %d: %s"%(len(entities),entities))

# Train the model with presets.
if MODE == 1:
    eval_metrics = train_model(train_data,entities,test_data=test_data,output_dir=output_dir,n_iter=int(n_iter))
    print(eval_metrics)
elif MODE == 2:
    metrics = []
    for i in range(len(data_folds)):
        train_data, test_data = split_data(data_folds, i)
        eval_metrics = train_model(train_data,entities,test_data=test_data,n_iter=n_iter)
        metrics.append(eval_metrics)
    print(metrics)
elif MODE == 3:
    train_model(DATA,entities,output_dir=output_dir,n_iter=n_iter)