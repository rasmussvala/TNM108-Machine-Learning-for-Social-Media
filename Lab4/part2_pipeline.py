from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

# Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# For evaluation
import numpy as np

# training SVM classifier
from sklearn.linear_model import SGDClassifier

# SVM more detailed performance analysis of the results:
from sklearn import metrics

# Parameter tuning using grid search
from sklearn.model_selection import GridSearchCV

# moviedir = r'C:\Users\Wille\Documents\GitHub\TNM108\Lab4\movie_reviews' # wille's computer
moviedir = r"Lab4\movie_reviews"  # rasmus' computer

# loading all files.
movie = load_files(moviedir, shuffle=True)

# Get the training and test in data and target
(
    movie_data_train,
    movie_data_test,
    movie_target_train,
    movie_target_test,
) = train_test_split(movie.data, movie.target, test_size=0.20, random_state=12)

# ALT 1: MultinomialNB

# # create the pipeline
# movie_clf = Pipeline(
#     [
#         ("vect", CountVectorizer()),
#         ("tfidf", TfidfTransformer()),
#         ("clf", MultinomialNB()),
#     ]
# )

# movie_clf.fit(movie_data_train, movie_target_train)
# predicted = movie_clf.predict(movie_data_test)
# print("multinomialBC accuracy ", np.mean(predicted == movie_target_test))

# ALT 2: SVM

# training SVM classifier
movie_clf = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        (
            "clf",
            SGDClassifier(
                loss="hinge",
                penalty="l2",
                alpha=1e-3,
                random_state=42,
                max_iter=5,
                tol=None,
            ),
        ),
    ]
)

movie_clf.fit(movie_data_train, movie_target_train)
predicted = movie_clf.predict(movie_data_test)
print("SVM accuracy ", np.mean(predicted == movie_target_test))

# ------------------ gridSearch ------------------

# we create possible parameters
parameters = {
    "vect__ngram_range": [(1, 1), (1, 2)],
    "tfidf__use_idf": (True, False),
    "clf__alpha": (1e-2, 1e-3),
}

# we gridsearch the best parameters
gs_clf = GridSearchCV(movie_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(movie_data_train, movie_target_train)

# print the best score
print("\ngrid_search_clf.best_score_:", gs_clf.best_score_, "\n")

# print the best parameters
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

# ------------------ predict movie reviews using gridsearch ------------------

# Fake movie reviews
reviews_new = [
    "This movie was excellent",
    "Absolute joy ride",
    "Steven Seagal was terrible",
    "Steven Seagal shone through.",
    "This was certainly a movie",
    "Two thumbs up",
    "I fell asleep halfway through",
    "We can't wait for the sequel!!",
    "!",
    "?",
    "I cannot recommend this highly enough",
    "instant classic.",
    "Steven Seagal was amazing. His performance was Oscar-worthy.",
]

# predict the target of the fake movie reviews
pred = gs_clf.predict(reviews_new)

# print the guesses of our fake movie reviews
print("\n")
for review, category in zip(reviews_new, pred):
    print("%r => %s" % (review, movie.target_names[category]))
