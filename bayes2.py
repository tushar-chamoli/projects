import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Step 3: Define the Naive Bayes classifier function
class NaiveBayes:
    
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.class_word_counts = {}
        self.class_total_counts = {}
        self.prior_probs = {}
    
    def train(self, X, y):
        # Transform text data into numerical vectors
        X_vectors = self.vectorizer.fit_transform(X)
        
        # Get the vocabulary from the vectorizer
        self.vocab = self.vectorizer.get_feature_names_out()
        
        # Compute the prior probabilities for each class
        self.prior_probs = {c: np.log(len(y[y == c]) / len(y)) for c in np.unique(y)}
        
        # Compute the word counts for each class
        self.class_word_counts = {c: {word: 0 for word in self.vocab} for c in np.unique(y)}
        self.class_total_counts = {c: 0 for c in np.unique(y)}
        for c in np.unique(y):
            X_c = X_vectors[y == c]
            self.class_word_counts[c] = np.asarray(X_c.sum(axis=0)).flatten()
            self.class_total_counts[c] = self.class_word_counts[c].sum()
                    
    def predict(self, X):
        # Transform test data into numerical vectors
        X_vectors = self.vectorizer.transform(X)
        
        y_pred = []
        for i in range(X_vectors.shape[0]):
            sentence_vector = np.asarray(X_vectors[i].todense()).flatten()
            
            # Compute the log probabilities for each class
            log_probs = {c: self.prior_probs[c] for c in self.class_word_counts}
            for j, word in enumerate(self.vocab):
                if sentence_vector[j] > 0:
                    for c in self.class_word_counts:
                        log_probs[c] += np.log((self.class_word_counts[c][j] + 1) / (self.class_total_counts[c] + len(self.vocab)))
            
            # Determine the predicted class
            y_pred.append(max(log_probs, key=log_probs.get))
        return y_pred
