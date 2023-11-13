import sys, os
os.system('cls')
import sklearn
from sklearn.datasets import load_files
# import CountVectorizer, nltk
from sklearn.feature_extraction.text import CountVectorizer
import nltk
#nltk.download('punkt')

# Convert raw frequency counts into TF-IDF (Term Frequency -- Inverse Document Frequency) values
from sklearn.feature_extraction.text import TfidfTransformer
# Split data into training and test sets
from sklearn.model_selection import train_test_split
# Now ready to build a classifier. 
# We will use Multinominal Naive Bayes as our model
from sklearn.naive_bayes import MultinomialNB
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# For grid and pipe
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import SGDClassifier
import numpy as np





# loading all files. 
moviedir = r'C:\Users\Wille\Documents\GitHub\TNM108\Lab4\movie_reviews'
movie = load_files(moviedir, shuffle=True)

#len(movie.data)

# target names ("classes") are automatically generated from subfolder names
#movie.target_names

# First file seems to be about a Schwarzenegger movie. 
#movie.data[0][:500]

# first file is in "neg" folder
#movie.filenames[0]

# first file is a negative review and is mapped to 0 index 'neg' in target_names
#movie.target[0]

# Three tiny "documents"
#docs = ['A rose is a rose is a rose is a rose.',
#        'Oh, what a fine day it is.',
#        "A day ain't over till it's truly over."]

        # Initialize a CountVectorizer to use NLTK's tokenizer instead of its 
#    default one (which ignores punctuation and stopwords). 
# Minimum document frequency set to 1. 
#fooVzer = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize)

# .fit_transform does two things:
# (1) fit: adapts fooVzer to the supplied text data (rounds up top words into vector space) 
# (2) transform: creates and returns a count-vectorized output of docs
#docs_counts = fooVzer.fit_transform(docs)

# fooVzer now contains vocab dictionary which maps unique words to indexes
#fooVzer.vocabulary_

# docs_counts has a dimension of 3 (document count) by 16 (# of unique words)
#docs_counts.shape

# this vector is small enough to view in a full, non-sparse form! 
#docs_counts.toarray()

#fooTfmer = TfidfTransformer()

# Again, fit and transform
#docs_tfidf = fooTfmer.fit_transform(docs_counts)

# TF-IDF values
# raw counts have been normalized against document length, 
# terms that are found across many docs are weighted down ('a' vs. 'rose')
#docs_tfidf.toarray()

# A list of new documents
#newdocs = ["I have a rose and a lily.", "What a beautiful day."]

# This time, no fitting needed: transform the new docs into count-vectorized form
# Unseen words ('lily', 'beautiful', 'have', etc.) are ignored
#newdocs_counts = fooVzer.transform(newdocs)
#newdocs_counts.toarray()

# Again, transform using tfidf 
#newdocs_tfidf = fooTfmer.transform(newdocs_counts)
#newdocs_tfidf.toarray()

movie_data_train, movie_data_test, movie_target_train, movie_target_test = train_test_split(movie.data, movie.target, 
                                                          test_size = 0.20, random_state = 12)

# # initialize CountVectorizer
# movieVzer= CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features=3000) # use top 3000 words only. 78.25% acc.
# # movieVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)         # use all 25K words. Higher accuracy

# # fit and tranform using training text 
# docs_train_counts = movieVzer.fit_transform(docs_train)

# # 'screen' is found in the corpus, mapped to index 2290
# movieVzer.vocabulary_.get('screen')

# # Likewise, Mr. Steven Seagal is present...
# movieVzer.vocabulary_.get('seagal')

# # huge dimensions! 1,600 documents, 3K unique terms. 
# docs_train_counts.shape

# # Convert raw frequency counts into TF-IDF values
# movieTfmer = TfidfTransformer()
# docs_train_tfidf = movieTfmer.fit_transform(docs_train_counts)

# # Same dimensions, now with tf-idf values instead of raw frequency counts
# docs_train_tfidf.shape

# # Using the fitted vectorizer and transformer, tranform the test data
# docs_test_counts = movieVzer.transform(docs_test)
# docs_test_tfidf = movieTfmer.transform(docs_test_counts)

# # Train a Multimoda Naive Bayes classifier. Again, we call it "fitting"
# clf = MultinomialNB()
# clf.fit(docs_train_tfidf, y_train)

# # Predict the Test set results, find accuracy
# y_pred = clf.predict(docs_test_tfidf)
# sklearn.metrics.accuracy_score(y_test, y_pred)

# cm = confusion_matrix(y_test, y_pred)
# cm

# pipeline ===============================================================
movie_clf = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)),
])
movie_clf.fit(movie_data_train, movie_target_train)
predicted = movie_clf.predict(movie_data_test)

print("SVM accuracy ",np.mean(predicted == movie_target_test))
print(metrics.classification_report(movie_target_test, predicted, target_names=movie.target_names))

print(metrics.classification_report(movie_target_test, predicted, target_names=movie.target_names))



# Parameter tuning using grid search
parameters = {
 'vect__ngram_range': [(1, 1), (1, 2)],
 'tfidf__use_idf': (True, False),
 'clf__alpha': (1e-2, 1e-3),
}
gs_clf = GridSearchCV(movie_clf, parameters, cv=5, n_jobs=-1)

grid_search_clf = gs_clf.fit(movie_data_train[:400], movie_target_train[:400])

# ======== ===============================================================

# very short and fake movie reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride', 
            'Steven Seagal was terrible', 'Steven Seagal shone through.', 
              'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through', 
              "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough', 
              'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']

#reviews_new_counts = movieVzer.transform(reviews_new)         # turn text into count vector
#reviews_new_tfidf = movieTfmer.transform(reviews_new_counts)  # turn into tfidf vector

print("\n\n\n============= Print out results grid =============")
for param_name in sorted(parameters.keys()):
 print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

# have classifier make a prediction
pred = gs_clf .predict(reviews_new)

# print out results
print("\n\n\n============= Print out results =============")
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))

