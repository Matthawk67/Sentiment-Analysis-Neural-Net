"""
*   test.py - 12/4/2018
*   Leo Neat #1452487
*
*   This is used to test a pre trained model on the imdb data set for cmps 142
*
*   Usage: python3 test.py -incsv <name of input file> -outcsv <name of output file> -mname <saved trained model name>
*
*   Package requirements: tensorflow, numpy, pandas, sklearn, (Optional: CUDA, CUDNN, tensorflow-gpu)
*
"""
import argparse
import pickle
from os import path
import pandas as pd
from keras import models
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import csv

FILE_LOC = 'data'

# Model parameters
SIZE_OF_BAG = 20000                 # The size of the bag used in the embedding
PADDED_LENGTH = 50                  # The amout of padding given to each input review


def pre_processing(csv_data_path):
    """
    Function to process the data converting each word to lower case and removing blank lines
    :returns: list of reviews
    """
    csv_data = pd.read_csv(csv_data_path)
    filtered_reviews = []
    for s in csv_data["Phrase"]:
        s = s.lower()                       # convert reviews to lower case
        words = s.split(' ')                # Tokenize them
        filtered = []
        for w in words:
            filtered.append(w)
        filtered_reviews.append(filtered)     # Add review that length is not zero
    return filtered_reviews


def bag_of_words(in_array, tokenizer):
    """
    Converts turns an array of reviews into a bag of words model, each word is represented by an integer
    :param in_array: Array of all of the reviews needed to be converted in to the bag of words model
    :param tokenizer:  A object to convert each word to an integer
    :return: a list of all the reviews padded to the correct length to be used for input into the model
    """
    word_list = []
    for word in in_array:
        word_list.append(word)
    tokenizer.fit_on_texts(word_list)
    tokened_reviews = []
    for review in in_array:
        sequ = tokenizer.texts_to_sequences(review)
        filtered_sequ = [x for x in sequ if x != []]
        to_pad = []
        for l in filtered_sequ:
            to_pad.append(l[0])
        tokened_reviews.append(pad_sequences([to_pad], PADDED_LENGTH))
    tokened_reviews_np = np.vstack(tokened_reviews)
    return tokened_reviews_np


def load_model(model_name, tokenizer_name):
    """
    This is a function to load the required objects to run inference on the model
    :param model_name: the name of the hdf5 model where the model is stored
    :param tokenizer_name: the name of the tokenizer used for the bag of words in defining this model
    :return: [keras model used for inference, tokenizer used for converting]
    """
    # loading tokeizer
    with open(tokenizer_name, 'rb') as handle:
        tokenizer = pickle.load(handle)

    model = models.load_model(model_name)
    return model, tokenizer


def write_csv_out(predictions, csv_data_path, out_csv):
    """
    This function is used to write the output predictions based on the specifications defined in the project description
    :param predictions: A list of predictions
    :param csv_data_path: The location ot the the test data
    :param out_csv: The path of the output location
    :return: None
    """
    csv_data = pd.read_csv(csv_data_path)
    phraseIds = []
    for s in csv_data['PhraseId']:
        phraseIds.append(s)

    with open(out_csv, 'w') as csvfile:
        fieldnames = ['PhraseId', 'Sentiment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(0, len(phraseIds)):
            writer.writerow({'PhraseId': str(phraseIds[i]), 'Sentiment': str(predictions[i])})
    print("CSV writing finished. Process complete.")


def main():
    # Check the user input arguments
    parser = argparse.ArgumentParser(description='Train a neural net on IMBD data')
    parser.add_argument('-incsv', type=str, nargs=1, help="The location of the csv input you want to run your model on", required=True)
    parser.add_argument('-mname', type=str, nargs=1, help="The name of the saved model you want to load", required=True)
    parser.add_argument('-outcsv', type=str, nargs=1, help="The name of the output csv file", required=True)
    args = parser.parse_args()

    # load model and tokenizer
    model_name = path.join(FILE_LOC, args.mname[0] + 'model.hdf5')
    tokenizer_name = path.join(FILE_LOC, args.mname[0] + 'tokenizer.pickle')
    model, tokenizer = load_model(model_name, tokenizer_name)

    # Read input CSV
    csv_data_path = path.join(FILE_LOC, args.incsv[0])
    reviews_list = pre_processing(csv_data_path)
    print("CSV read " + str(len(reviews_list)) + " sentences")
    tokend_reviews = bag_of_words(reviews_list, tokenizer)
    print("Starting prediction of " + str(len(tokend_reviews)) + " sentences")
    predictions = model.predict_classes(tokend_reviews)
    print("Finished predicting " + str(len(predictions)) + " values")

    # Write output
    out_csv = path.join(FILE_LOC, args.outcsv[0])
    print("Writing result csv to " + out_csv)
    write_csv_out(predictions, csv_data_path, out_csv)


if __name__ == "__main__":
    main()
