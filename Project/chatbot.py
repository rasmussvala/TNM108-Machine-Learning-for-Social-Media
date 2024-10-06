import pandas as pd
import re
import os  # to clear console
from functions import text_normalization, chat_tfidf
from nltk.corpus import stopwords  # for stop words
import nltk

nltk.download("stopwords")
os.system("cls")  # clears console

# Initialize df from excel file, needs bye
print("File ID (1, 2, 3 or 4): ", end="")
file_id = input()
print("\n\nChatbot conversation using data from ", end="")
if file_id == "1":
    data_frame = pd.read_excel("datasets/humanConversation.xlsx")
    print("human conversation")
elif file_id == "2":
    data_frame = pd.read_excel("datasets/3KConversation.xlsx")
    print("3k conversation")
elif file_id == "3":
    data_frame = pd.read_excel("datasets/topicalChatFiltered.xlsx")
    print("topical chat")
else:
    data_frame = pd.read_excel("datasets/allConversations.xlsx")
    print("all conversations")
print("================================")

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
