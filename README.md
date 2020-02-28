# Similarity-score-of-an-input-string-with-multiple-documents

In this python script, several text documents are read and stored. Then with these along with an input text string (i.e. used to test the similarity), a train set is created.

The train set is then vectorised using the **Tfidf Vectoriser** which also removes the stopwords. 

Hence, the similarity score for the two vectors is calculated using Scikit learn's **cosine_similarity**. Cosine similarity is a metric used to measure how similar the documents are irrespective of their size. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space. The cosine similarity is advantageous because even if the two similar documents are far apart by the Euclidean distance (due to the size of the document), chances are they may still be oriented closer together. *The smaller the angle, higher the cosine similarity.*
