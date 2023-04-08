from sklearn.linear_model import LogisticRegression
import numpy as np

# Adjust DATA for NLP

# Define the data and labels
data = [
  ["I want to add", 1],
  ["addition please", 1],
  ["let me plus something", 1],
  ["sum these numbers", 1],
  ["total of", 1],
  ["add these up", 1],
  ["combine these values", 1],
  ["add the following", 1],
  ["add these numbers together", 1],
  ["find the sum of these numbers", 1],
  ["what's the total of these numbers", 1],
  ["please add these numbers", 1],
  ["I need to subtract", 2],
  ["minus", 2],
  ["take away from", 2],
  ["subtract these values", 2],
  ["remove this number from", 2],
  ["subtract this from that", 2],
  ["what's the difference between these numbers", 2],
  ["please subtract these numbers", 2],
  ["find the result of subtracting these numbers", 2],
  ["I want multiplication", 3],
  ["times please", 3],
  ["multiply by", 3],
  ["what's the product of", 3],
  ["calculate the product of", 3],
  ["multiply these values", 3],
  ["double this number", 3],
  ["triple this number", 3],
  ["quadruple this number", 3],
  ["multiply these numbers together", 3],
  ["find the result of multiplying these numbers", 3],
  ["please multiply these numbers", 3]
]

sentences = [d[0] for d in data]
labels = [d[1] for d in data]

# Define the vectorizer and model
class CustomCountVectorizer:
    def __init__(self):
        self.vocab = {}
        self.num_docs = 0

    def fit_transform(self, docs):
        self.vocab = {}
        self.num_docs = 0
        for doc in docs:
            self.num_docs += 1
            for word in doc.split():
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

        X = np.zeros((len(docs), len(self.vocab)))
        for i, doc in enumerate(docs):
            for word in doc.split():
                X[i][self.vocab[word]] += 1

        return X

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, doc in enumerate(docs):
            for word in doc.split():
                if word in self.vocab:
                    X[i][self.vocab[word]] += 1

        return X

vectorizer = CustomCountVectorizer()
model = LogisticRegression()

X = vectorizer.fit_transform(sentences)
model.fit(X, labels)

def classify_input(input_text):
    X_test = vectorizer.transform([input_text])
    prediction = model.predict(X_test)
    return prediction[0]

print(classify_input("let's add these numbers"))

print(classify_input("subtract this value"))

print(classify_input("multiply these values"))