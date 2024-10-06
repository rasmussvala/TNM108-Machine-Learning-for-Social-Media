# TNM108 - Machine Learning for Social Media
 
This repository is a part of the course [TNM108 Machine Learning for Social Media](https://studieinfo.liu.se/en/kurs/TNM108/ht-2024) at Linköping University 2024. It contains both labs and a project.

## Project - Chatbot

This project is a simple conversational chatbot implemented in Python, which runs directly in the terminal. It allows users to engage in text-based conversations, using pre-processed conversational datasets for its responses.

### Datasets

The chatbot is powered by multiple large conversational datasets, including:

- [3K Conversations Dataset for ChatBot](https://www.kaggle.com/datasets/kreeshrajani/3k-conversations-dataset-for-chatbot)
- [Human Conversation training data](https://www.kaggle.com/datasets/projjal1/human-conversation-training-data)
- [Chatbot Dataset Topical Chat](https://www.kaggle.com/datasets/arnavsharmaas/chatbot-dataset-topical-chat)

### How It Works

The chatbot uses TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity to process user input and select the most relevant response from the pre-processed dataset. Here's an overview: 

- **Preprocessing**: Datasets' are lemmatized and features are extracted using TF-IDF.
- **Similarity Matching**: User input is compared with the dataset using cosine similarity to find the closest matching response.
- **Response**: The chatbot returns the most relevant response based on the similarity score.

### Installation

Clone the repository:
```
git clone https://github.com/rasmussvala/TNM108-Machine-Learning-for-Social-Media.git
```

Install required dependencies (bonus point if you do it in a virtual environment):
```
pip install -r requirements-chatbot.txt
```

### Usage
Run the chatbot in the terminal:

```
py chatbot.py
```
