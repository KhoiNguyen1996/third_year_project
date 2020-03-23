# Function to train a custom Name Entity Recognition spaCy model.
# Copied from spaCy website and modified.
# https://spacy.io/usage/training

from __future__ import unicode_literals, print_function
import plac
import random
import spacy
activated = spacy.prefer_gpu()

from pathlib import Path
from spacy.util import minibatch, compounding

# Function to evaluate model Precision, Recall and F-Score metric.
# This method assume data is in spaCy format.
# Input model and data to evaluate.
# Output model accuracy metrics.
def evaluate_model(nlp, data, entities, verbose=False):
    prediction_count = 0
    gold_count = 0
    interception_count = 0

    # Setup entity transition matrix for evaluation.
    label_intercept = {}
    label_gold = {}
    label_pred = {}
    for label in entities:
        label_intercept[label] = 0
        label_gold[label] = 0
        label_pred[label] = 0

    # Estimate prediction accuracy
    for value in data:
        preds = {}
        ents = {}

        # Get sentence string
        sentence = value[0]

        # Add prediction entities
        docs = nlp(sentence)
        for ent in docs.ents:
            preds[ent.text]=ent.label_
            label_pred[ent.label_] = label_pred[ent.label_] + 1
        prediction_count = prediction_count + len(preds)

        # Add gold entities
        sentence_ents = value[1]['entities']
        for ent in sentence_ents:
            start = ent[0]
            end = ent[1]
            ent_type = ent[2]
            ents[sentence[start:end]]=ent_type
            label_gold[ent_type] = label_gold[ent_type] + 1
        gold_count = gold_count + len(ents)

        if ents==preds:
            interception_count = interception_count + len(ents)
            for word in ents:
                label_intercept[ents[word]]=label_intercept[ents[word]]+1
        else:
            for word in preds:
                if word in ents:
                    if ents[word] == preds[word]:
                        interception_count = interception_count + 1
                        label_intercept[ents[word]]=label_intercept[ents[word]]+1

    # Finished evaluating and print accuracy metric.
    precision = interception_count/prediction_count
    recall = interception_count/gold_count
    f_score = (2.0 * precision * recall) / (precision + recall)
    
    # Variable to store evaluation data of each label
    label_results = {}

    for label in entities:
        pre_scr = float(label_intercept[label]/label_pred[label])
        rec_scr = float(label_intercept[label]/label_gold[label])

        # In case no label detected.
        if (pre_scr + rec_scr) == 0:
            f_scr = 1
        else:
            f_scr = (2.0 * pre_scr * rec_scr) / (pre_scr + rec_scr)

        if verbose:
            print("Label-%s accuracy {precision: %f, recall: %f, f-score: %f}"%(label,pre_scr, rec_scr, f_scr))

    print("Model accuracy {precision: %f, recall: %f, f-score: %f}"%(precision, recall, f_score))
    return(precision,recall,f_score)

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)

def train_model(train_data,entities,
                test_data=None,
                model=None,
                output_dir=None,
                n_iter=10):
    """1. Load the model if user input any or create a blank model
       2. Set up the name entity recognition pipeline and train the entity recognizer.
       3. Add new label entity and initialise model.

       Input: train_data, entities label.
       Output: Trained Natural Language Processing model."""

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        # Building a blank model for training.
        nlp=spacy.blank('en')
        nlp.vocab.vectors.name='adr_vocab'   
        ner=nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

        # Add new labels.
        for entity in entities:
            nlp.entity.add_label(entity)
        print("Created blank 'en' model")

    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Add new/current labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Disable other pipelines during training, except
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # Only train name entity recognition pipe.
    with nlp.disable_pipes(*other_pipes):
        if model is None:
            optimizer = nlp.begin_training()
        else:
            optimizer = nlp.resume_training()

        # Shuffle our training data before training
        random.shuffle(train_data)
        test_results = []

        for itn in range(n_iter):
            print("SpaCy Model training epoch: %d"%(itn+1))
            # Shuffle the data for each training iteration
            random.shuffle(train_data)

            losses = {}
            # Batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    sgd=optimizer, # optimizers
                    drop=0.2,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

            # Evaluate the model after training each data partition
            if test_data is not None:
                #try:
                result = evaluate_model(nlp, test_data, entities)
                test_results.append(result)
                #except:
                #    pass

    # Model finished training
    # Save output to directory.
    if output_dir is not None:
        # If output directory not existed create one.
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        # Save model to disk
        nlp.to_disk(output_dir)
        # Debug log infomation
        print("Saved model to", output_dir)
        print("Model finished training.")
    
    return(test_results)
