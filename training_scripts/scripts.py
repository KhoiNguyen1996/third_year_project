import spacy
activated = spacy.prefer_gpu()

import pickle
import datetime
import random
import numpy as np
import pandas as pd
from spacy.scorer import Scorer
from spacy import displacy
from spacy.gold import GoldParse
from spacy.gold import biluo_tags_from_offsets

def is_unit_tag(tag):
    tag_type = tag[:2]
    stand_alone = ["O","U-"]
    if tag_type in stand_alone:
        return(True)
    else:
        return(False)

# Function to check iob tags.
def check_iob(iob):
    stack = []
    verbose = []
    for label in iob:
        tag = label[:2]
        word = label[2:]
        verbose.append(label)
        # Tag in stand alone and stack is empty. 1 1
        if is_unit_tag(tag) and len(stack) == 0:
            pass
        # Tag not in stand alone and stack not empty. 0 1
        elif not is_unit_tag(tag) and len(stack) == 0:
            if not tag == "B-":
                return(False)
            else:
                stack.append(label)
        # Tag in stand alone and stack not empty. 1 0
        elif is_unit_tag(tag) and not len(stack) == 0:
            return(False)
        # Tag not in stand alone and not empty. 0 0 
        else:
            if tag == "B-" and stack[-1][2:] == word:
                return(False)
            elif tag == "I-" and stack[-1][2:] == word:
                stack.append(label)
            elif tag == "L-" and stack[-1][2:] == word:
                stack = []
            else:
                print(iob)
                print("Unknown error iob tags:\n%s\n%s"%(verbose,stack))
                return(False)
        #print(stack)
    return(True)

def convertspacyapiformattocliformat(nlp, TRAIN_DATA):
    docnum = 1
    documents = []
    iob_tags = []
    errors = {}
    for t in TRAIN_DATA:
        doc = nlp(t[0])
        tags = biluo_tags_from_offsets(doc, t[1]['entities'])
        for i in range(len(tags)):
            if tags[i] == "-":
                tags[i] = tags[i].replace("-","O")
        # Check if tokens are correct else don't append.
        if check_iob(tags):
            ner_info = list(zip(doc, tags))
            tokens = []
            sentences = []
            for n, i in enumerate(ner_info):
                token = {"head" : 0,
                "dep" : "",
                "tag" : "",
                "orth" : i[0].string,
                "ner" : i[1],
                "id" : n}
                tokens.append(token)
            iob_tags.append([[token.text for token in doc],tags])
            sentences.append({'tokens' : tokens})
            document = {}
            document['id'] = docnum
            docnum+=1
            document['paragraphs'] = []
            paragraph = {'raw': doc.text,"sentences" : sentences}
            document['paragraphs'] = [paragraph]
            documents.append(document)
        else:
            print(t[0])
            print("ERRORS BILOU TAGS:\n%s"%(tags))
            errors[t[0]] = tags
    return(documents, iob_tags, errors)

# Function to get nth value in a dictionary.
def dict_subset(dict_data,index):
    count = 0
    result = []
    for key in dict_data:
        if count < index:
            sentence = key[0]
            ents = key[1]
            result.append([sentence,ents])
        else:
            break
        count = count + 1
    return(result)

# Split 10-fold data for model cross-validation.
def partition_data(data):
    length = int(len(data)/10)
    folds = []
    for i in range(9):
        folds += [data[i*length:(i+1)*length]]
    folds += [data[9*length:len(data)]]
    return(folds)

# Get index of a data partition fold.
def split_data(data_folds,index):
    # Splitting data to train and test sets.
    test_data = data_folds[index]
    train_data = []

    for fold in range(len(data_folds)):
        if fold != index:
            train_data.extend(data_folds[fold])

    return(train_data,test_data)

# Function to get current time in UTC format as string
def get_time():
    utc_datetime = datetime.datetime.utcnow()
    time = utc_datetime.strftime("%Y-%m-%d-%H%M%S")
    return(time)

# Save a data object as pickle file
def save_pkl(pkl_name, data):
    pickle_file = open(pkl_name, 'wb')
    pickle.dump(data, pickle_file)
    print("Writing data is done!")

# Load a pickle data object
def load_pkl(pkl_name):
    pickle_file = open(pkl_name, 'rb') 
    return(pickle.load(pickle_file))

# Function to match a word to a sentence
# Return start, end index and a NER tag.
def match_word(sentence, word, tag):
    if word not in sentence:
        print("*************ERROR*******************")
        print(sentence)
        print(word)
        print("*************NO MATCHED**************")
        pass
    else:
        sentence.find(word)
        start_index = sentence.find(word)
        end_index = start_index + len(word)
        return(start_index, end_index, tag)

# Function to check if entry exist
# Combine many entity and a data tuple.
def add_entity(sentence, *args):
    sentence_dict = {}
    sentence_dict['entities'] = []
    if isinstance(sentence, tuple):
        sent = sentence[0]
        ent_dict = sentence[1]
        for entity in args:
            if entity is not None and entity not in ent_dict:
                ent_dict['entities'].append(entity)
        return(sent,ent_dict)
    else:
        for entity in args:
            if entity is not None and entity not in sentence_dict['entities']:
                sentence_dict['entities'].append(entity)
    return(sentence,sentence_dict)

# Model evaluation with all the entities
def evaluate(ner_model, examples):
    scorer = Scorer()
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot['entities'])
        pred_value = ner_model(input_)

        try:
            scorer.score(pred_value, gold)
        except:
            print("***************Error***************")       
            print(input_)
            print("***************Skipped*************")
            continue
    return scorer.scores

# Model evaluation with respect to an entity.
def evaluate_entity(nlp, examples, ent='DRUG'):
    scorer = Scorer()
    for input_, annot in examples:
        text_entities = []
        for entity in annot.get('entities'):
            if ent in entity:
                text_entities.append(entity)
        doc_gold_text = nlp.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=text_entities)
        pred_value = nlp(input_)
        scorer.score(pred_value, gold)
    return scorer.scores

# Testing functions
def get_data(raw_data, index):
    smp=index
    print("***************Sentence***************\n %s"%raw_data["sentence"].iloc[smp])
    print("*****************Drug*****************\n %s"%(raw_data["DRUG"].iloc[smp]))

    print("*****************Data*****************")
    ent_0=match_word(raw_data["sentence"].iloc[smp], raw_data["DRUG"].iloc[smp], "DRUG")
    spacy_data_0=add_entity(raw_data["sentence"].iloc[smp], ent_0)
    print(spacy_data_0)


# Function to convert BILOU to IOB tags.
# Convert all tags from L to I and all U to B.
# Get rid of '-' tags error.
def bilou_to_iob(tags):
    words = [tag.replace('L-', 'I-') for tag in tags]
    words = [tag.replace('U-', 'B-') for tag in words]
    words = [tag.replace('-', 'O') for tag in words]
    words = [tag.replace('IO', 'I-') for tag in words]
    tokens = [tag.replace('BO', 'B-') for tag in words]
    
    return(tokens)

# Given a column data return histogram values.
def get_freq_stats(df_column):
    # Frequency of different tags in the data
    tags_list=df_column.unique()
    tags_dict={}
    for tag in tags_list:
        tag_df=df_column[df_column==tag]
        print("Tags %s in the data: %d"%(tag, len(tag_df)))
        tags_dict[tag]=len(tag_df)
    return(tags_dict)
