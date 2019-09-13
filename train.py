"""
*   train.py - 12/4/2018
*   Leo Neat #1452487
*
*   This is used to train different type of neural networks on the IMDB data set for sentiment analysis
*   If you do not have a name flag the model will not save
*
*   Usage: python3 train.py -mtype <model type> -name <saved trained model name>
*
*   Package requirements: tensorflow, numpy, pandas, sklearn, (Optional: CUDA, CUDNN, tensorflow-gpu)
*
"""
import numpy as np
import sys
import argparse
import pickle
import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer

from keras.regularizers import l1, l2
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, MaxPooling1D, Conv1D, Dropout, BatchNormalization, Flatten

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Use GPU for training
K.tensorflow_backend._get_available_gpus()

# Consts
DATA_LOC = "data"                   # Folder where the default train data is
TRAIN_NAME = "train.csv"            # Folder where the default test data is

# Model parameters
SIZE_OF_BAG = 20000                 # The size of the bag used in the embedding
PADDED_LENGTH = 50                  # The amout of padding given to each input review
SIZE_OF_WORDVEC = 64                # The length each word vector is given in the model
K = 10                              # Number used for cross validation of models

def pre_processing():
    """
    Function to process the data converting each word to lower case and removing blank lines
    :returns: [list of reviews, list of sentiments]
    """
    csv_data = pd.read_csv(os.path.join(DATA_LOC, TRAIN_NAME))
    counter = 0
    filtered_reviews = []
    filtered_sentiments = []
    for s in csv_data["Phrase"]:
        s = s.lower()                       # convert reviews to lower case
        words = s.split(' ')                # Tokenize them
        filtered = []
        for w in words:
            filtered.append(w)
        if len(filtered) > 0:
            filtered_reviews.append(filtered)     # Add review that length is not zero
            filtered_sentiments.append(csv_data["Sentiment"][counter])
        counter += 1
    return filtered_reviews, np.asarray(filtered_sentiments)


def bin_sentiment(sent):
    """
    Function to convert sentiment values in to a list of  length 5 to allow for the softmax function to work
    :param sent: The int representation of the sentiment
    :return: The binary list representation of the sentiment, used for the last layer of the neural net
    """
    encoder = LabelBinarizer()
    sent_np = encoder.fit_transform(sent)
    return sent_np


def create_modelLSTM():
    """
    Creates an LSTM keras model that can bet trained and then used for inference
    :return: keras model
    """
    model = Sequential()
    model.add(Embedding(SIZE_OF_BAG, SIZE_OF_WORDVEC, input_length=PADDED_LENGTH))  # Embed the word vectors to input nodes
    model.add(Dropout(0.5))                             # Drop out to increase generalization of model and reduce outfitting
    model.add(BatchNormalization())
    model.add(LSTM(32, kernel_regularizer=l2(0.001)))   # Primary layer of the model, an RNN
    model.add(Dropout(0.5))                             # Drop out to increase generalization of model and reduce outfitting
    model.add(Dense(5, activation='softmax'))           # Output layer to determine which class it is

    # Adadelta was used because it preformed the best and is good at working with sparse data
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


def create_modelCNN():
    """
    Creates an CNN keras model that can bet trained and then used for inference
    :return: keras model
    """
    model = Sequential()
    model.add(Embedding(SIZE_OF_BAG, SIZE_OF_WORDVEC, input_length=PADDED_LENGTH))  # Embed the word vectors to input nodes
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(128))
    model.add(Dropout(0.5))                             # Drop out to increase generalization of model and reduce outfitting
    model.add(Dense(5, activation='softmax'))           # Output layer to determine which class it is

    # Adadelta was used because it preformed the best and is good at working with sparse data
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_modelSTD():
    """
    Creates an standard keras model that can bet trained and then used for inference
    :return: keras model
    """
    model = Sequential()
    model.add(Embedding(SIZE_OF_BAG, SIZE_OF_WORDVEC, input_length=PADDED_LENGTH))  # Embed the word vectors to input nodes
    model.add(Dropout(0.2)) # Drop out to increase generalization of model and reduce outfitting
    model.add(Flatten())
    model.add(Dense(20, activation='tanh'))
    model.add(Dropout(0.5))  # Drop out to increase generalization of model and reduce outfitting
    model.add(Dense(20, activation='relu'))
    model.add(Dense(5, activation='softmax'))  # Output layer to determine which class it is

    # Adam optimizer was used instead of adadelta
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


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


def eval(model, input_features, results):
    """
    Evaluates the model based on the defined metric, used after training is stopeed
    :param model: The keras model that needs to be evaluated
    :param input_features: The features that need to be tested
    :param results:
    :return:
    """
    scores = model.evaluate(input_features, results)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def save_model(tokenizer, model, model_name):
    """

    :param tokenizer: The tokenizer object that is used to split up the bag of words
    :param model: The trained model object
    :param model_name: the string of the model name
    :return:
    """
    with open(os.path.join(DATA_LOC, model_name + "tokenizer.pickle"), "wb") as f:
        pickle.dump(tokenizer, f)
    model.save(os.path.join(DATA_LOC,model_name + "model.hdf5"))


def main():
    # Check the user input arguments
    parser = argparse.ArgumentParser(description='Train a neural net on IMBD data')
    parser.add_argument('-name', type=str, help="The name you want to save this model as: IMBD_TEST_1")
    parser.add_argument('-mtype', type=str, help="The model type for training: <LSTM, CNN, std>", required=True)
    args = parser.parse_args()

    model_name = args.name
    if model_name is not None:
        print("This model will be saved under the name: " + str(model_name))
    else:
        print("This model will be trained but not saved, add a name argument if you want it to be saved.")

    model_type = args.mtype.lower()
    tokenizer = Tokenizer(num_words=SIZE_OF_BAG)
    reviews, sentiments = pre_processing()
    bow_reviews = bag_of_words(reviews, tokenizer=tokenizer)

    # If the model is not being saved then it must be evaluated forperformance and testing purposes
    if model_name is None:
        # 10 fold cross validation of models
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        cvscores_acc = []
        cvscores_pre = []
        cvscores_rec = []
        cvscores_f1 = []
        for train, test in kfold.split(bow_reviews, sentiments):
            bin_sentiments = bin_sentiment(sentiments)
            if model_type == 'lstm':
                model = create_modelLSTM()
            elif model_type == 'cnn':
                model = create_modelCNN()
            elif model_type == 'std':
                model = create_modelSTD()
            else:
                print("Error you have an invalid model name please use: std, cnn, or lstm")
                sys.exit(1)
            model.fit(bow_reviews[train], bin_sentiments[train], epochs=25, batch_size=2000)
            scores = model.evaluate(bow_reviews[test], bin_sentiments[test], verbose=0)
            cvscores_acc.append(scores[1]*100)

            prf = precision_recall_fscore_support(sentiments[test], model.predict_classes(bow_reviews[test]))
            cvscores_pre.append(np.mean(prf[0]))
            cvscores_rec.append(np.mean(prf[1]))
            cvscores_f1.append(np.mean(prf[2]))
        print("The accuracy for each fold: " + str(cvscores_acc))
        print("The precision for each fold: " + str(cvscores_pre))
        print("The recall for each fold: " + str(cvscores_rec))
        print("The f1 for each fold: " + str(cvscores_f1))
        print("The average accuracy: " + str(np.mean(cvscores_acc)))
        print("The average precision: " + str(np.mean(cvscores_pre)))
        print("The average recall: " + str(np.mean(cvscores_rec)))
        print("The average f1: " + str(np.mean(cvscores_f1)))

    # If the user wants to save the model to use in inference at a later date
    else:
        bin_sentiments = bin_sentiment(sentiments)
        if model_type == 'lstm':
            model = create_modelLSTM()
        elif model_type == 'cnn':
            model = create_modelCNN()
        elif model_type == 'std':
            model = create_modelSTD()
        else:
            print("Error you have an invalid model name please use: std, cnn, or lstm")
            sys.exit(1)
        model.fit(bow_reviews, bin_sentiments, epochs=40, batch_size=1000)
        save_model(tokenizer, model, model_name)


if __name__ == "__main__":
    main()