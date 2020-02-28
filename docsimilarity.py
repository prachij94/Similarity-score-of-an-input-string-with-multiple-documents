#Import the required libraries

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Bring in standard stopwords
stopWords = stopwords.words('english')

print ("\nCalculating document similarity scores...")

# Open and read a bunch of files 
f = open('C:/Users/prachi/Downloads/excavators.txt')
doc1 = str(f.read())

f = open('C:/Users/prachi/Downloads/hydraulics.txt')
doc2 = str(f.read())

f = open('C:/Users/prachi/Downloads/rollers.txt')
doc3 = str(f.read())

# Create a string to use to test the similarity scoring

train_string = 'jcb excavator'

# Construct the training set as a list
train_set = [train_string, doc1, doc2, doc3]

# Set up the vectoriser, passing in the stop words
tfidf_vectorizer = TfidfVectorizer(stop_words=stopWords)

# Apply the vectoriser to the training set
tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)

# Print the score
print("\nSimilarity Score [*] ",cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train))

