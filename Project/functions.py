import nltk
import numpy as np
import random

nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("punkt_tab")
import re
import pandas as pd
from nltk.stem import wordnet  # to perform Lemmitzation
from sklearn.feature_extraction.text import CountVectorizer  # to perform bow
from sklearn.feature_extraction.text import TfidfVectorizer  # to perform tfidf
from nltk import pos_tag  # for parts of speech
from sklearn.metrics import pairwise_distances  # to perform cosine similarity
from nltk import word_tokenize  # to creat tokens


def text_normalization(text):
    # ------------ Remove uppercase and special char ------------
    # text to lower case
    text = str(text).lower()

    # removing everything not [a, z]
    spl_char_text = re.sub(r"[^ a-z]", "", text)

    # word tokenizing (creates an object for every word)
    tokens = nltk.word_tokenize(spl_char_text)

    # ------------ Lemmatization ------------
    # lemmatization means reducing the word to its canonical
    # dictionary form (i.e. improving -> improve)
    # POS - part-of-speech

    # initalizing lemmatization
    lema = wordnet.WordNetLemmatizer()

    # parts-of-speech
    tags_list = pos_tag(tokens, tagset=None)

    # initialize an empty list for lemmatized words
    lema_words = []

    # iterate through each token (word) and its associated part-of-speech (POS),
    # used for determining the lemmmatization of the word
    for token, pos_token in tags_list:
        if pos_token.startswith("V"):  # verb
            pos_val = "v"

        elif pos_token.startswith("J"):  # adjective
            pos_val = "a"

        elif pos_token.startswith("R"):  # adverb
            pos_val = "r"

        else:
            pos_val = "n"  # noun

        # perform lemmatization
        lema_token = lema.lemmatize(token, pos_val)

        # append (add) the lemmatized token (word) into a list
        lema_words.append(lema_token)

    # return the lemmatized tokens as a sentence
    return " ".join(lema_words)


def bag_of_words(question, data_frame_lemma):
    # WIP
    print("bag of words is a WIP")


def chat_tfidf(question, data_frame_lemma):
    lemma = text_normalization(question)
    tfidf = TfidfVectorizer()
    x_tfidf = tfidf.fit_transform(data_frame_lemma).toarray()
    question_tfidf = tfidf.transform([lemma]).toarray()
    data_frame_tfidf = pd.DataFrame(x_tfidf, columns=tfidf.get_feature_names_out())
    cos = 1 - pairwise_distances(data_frame_tfidf, question_tfidf, metric="cosine")

    # argsort to get indices for sorting
    cos_sorted_indices = np.argsort(cos, axis=None)[::-1]
    cos_sorted = np.sort(cos, axis=None)[::-1]
    best_answer_cos = cos_sorted[0]

    nr_of_possible_answers = 10
    answers_array_indices = []

    for i in range(nr_of_possible_answers):
        if best_answer_cos*0.8 <= cos_sorted[i]:
            answers_array_indices.append(cos_sorted_indices[i])
            # print("cos dist: ", cos_sorted[i])
    return random.choice(answers_array_indices)
