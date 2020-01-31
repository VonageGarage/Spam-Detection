%%writefile train/train.py

import argparse
import pandas as pd
import os
import glob
import io

from sklearn import tree
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()

def remove_stop_words(words):
    result = [i for i in words if i not in ENGLISH_STOP_WORDS]
    return result

def word_stemmer(words):
    return [stemmer.stem(o) for o in words]

def word_lemmatizer(words):
    return [lemmatizer.lemmatize(o) for o in words]
    
def remove_characters(words):
    return [word for word in words if len(word)> 1]

def clean_token_pipeline(words):
    cleaning_utils = [remove_stop_words, word_lemmatizer]
    for o in cleaning_utils:
        words = o(words)
    return words
    
def process_text(X_train, X_test, y_train, y_test):
    X_train = [word_tokenize(o) for o in X_train]
    X_test = [word_tokenize(o) for o in X_test]

    X_train = [clean_token_pipeline(o) for o in X_train]
    X_test = [clean_token_pipeline(o) for o in X_test]

    X_train = [" ".join(o) for o in X_train]
    X_test = [" ".join(o) for o in X_test]
    
    return X_train, X_test, y_train, y_test

def convert_to_feature(raw_tokenize_data):
    raw_sentences = [' '.join(o) for o in raw_tokenize_data]
    return vectorizer.transform(raw_sentences)

def _npy_loads(data):
    """
    Deserializes npy-formatted bytes into a numpy array
    """
    stream = io.BytesIO(data)
    return np.load(stream)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()
    
    print("files {}".format(glob.glob(args.train+"/*")))
    print("out")
    
    train_data = pd.read_csv(args.train+"/spam_ass-cleaned.csv", index_col=0)
    train_data.dropna(inplace=True)
    print(train_data.head())
    
    X_train, X_test, y_train, y_test = train_test_split(train_data['message'], train_data['label'], test_size = 0.2, random_state = 1)
    X_train, X_test, y_train, y_test = process_text(X_train, X_test, y_train, y_test)

    X_train = [o.split(" ") for o in X_train]
    X_test = [o.split(" ") for o in X_test]

    vectorizer = TfidfVectorizer()
    raw_sentences = [' '.join(o) for o in X_train]
    vectorizer.fit(raw_sentences)
    
    print("saving transformer to {}".format(args.model_dir))
    joblib.dump(vectorizer, os.path.join(args.model_dir, "vectorizer.joblib"))

    x_train_features = convert_to_feature(X_train)
    x_test_features = convert_to_feature(X_test)

    clf = GaussianNB()
    clf.fit(x_train_features.toarray(),y_train)
    
    y_true, y_pred = y_test, clf.predict(x_test_features.toarray())
    print(classification_report(y_true, y_pred))

    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))
    
def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("model loaded {}".format(clf))
    return clf

def input_fn(request_body, request_content_type):
    print("** input_fn**")
    print("request_body:{} request_content_type:{}".format(request_body, request_content_type))

    if request_content_type == "text/plain":
        #convert to string
        message = str(request_body)
        print("converting message to string {}".format(message))
        return message
    elif request_content_type == "application/x-npy":
        return " ".join(_npy_loads(request_body))
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        return request_body

def predict_fn(input_data, model):
    
    print("** predict_fn**")
    print("input_data: {} model:{}".format(input_data, model))

    prefix = '/opt/ml/'
    model_path = os.path.join(prefix, 'model')
    my_vect = joblib.load(os.path.join(model_path, "vectorizer.joblib"))
    
    message = "".join(clean_token_pipeline(input_data))
    print("processed message: {}".format(message))
    message = my_vect.transform([message])
    message = message.toarray()
    prediction = model.predict(message)
    return prediction

def output_fn(prediction, accept):        
    print('Serializing the generated output. {} {}'.format(prediction, accept))
    return prediction, accept
