import sklearn
from sklearn.datasets import load_files

moviedir = r"Lab4\movie_reviews"

# loading all files.
movie = load_files(moviedir, shuffle=True)

# import CountVectorizer, nltk
from sklearn.feature_extraction.text import CountVectorizer
import nltk
