import pandas as pd
import re
import os  # to clear console
from functions import text_normalization, chat_tfidf
from nltk.corpus import stopwords  # for stop words
import nltk
nltk.download("stopwords")
os.system("cls")  # clears console

# Initialize df from excel file, needs bye
print("File ID: ", end="")
file_id = input()
print("\n\nChatbot conversation")
print("================================")
if file_id == 0:
    data_frame = pd.read_excel("conversations.xlsx")
else:
    data_frame = pd.read_excel("conversations.xlsx")

# Main code
data_frame["lemmatized_text"] = data_frame["question"].apply(text_normalization)

# all the stop words we have
stop_words = stopwords.words("english")
# print(stop_words)
bye = False

while not bye:
    # user input
    print("You: ", end="")
    user_question = input()

    user_question = user_question.lower()
    user_question = re.sub(r"[^ a-z]", "", user_question)

    # it the user writes a bye frase
    if (
        user_question == "bye"
        or user_question == "good bye"
        or user_question == "im out"
    ):
        bye = True

    # creates an empty array of stopwords removed
    stop_words_removed = []
    # split the question into an array of words
    words = user_question.split()

    # loop all words from the question
    for word in words:
        if word in stop_words:
            continue
        else:
            stop_words_removed.append(word)

        processed_question = " ".join(stop_words_removed)

    index_answer = chat_tfidf(processed_question, data_frame["lemmatized_text"])
    # print(index_value)
    print("ChatBoi:", data_frame["answer"].loc[index_answer])
    
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")

    # Fit and transform the vectorizer on the given sentence
    tfidf_matrix = tfidf_vectorizer.fit_transform([data_frame["answer"].loc[index_answer]])

    # Get the feature names (words) from the vectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Find the word with the highest TF-IDF score
    max_tfidf_index = tfidf_matrix.argmax()
    keyword = feature_names[max_tfidf_index]
    print(keyword)
