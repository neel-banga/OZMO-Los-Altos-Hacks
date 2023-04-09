from sklearn.linear_model import LogisticRegression
import numpy as np
import random

data = [
    ["Let me send a text to confirm the plans.", 0],
    ["Send an email to John asking him to call me.", 0],
    ["Read my latest emails.", 0],
    ["Text mom and ask her to pick me up.", 0],
    ["Read my unread messages on WhatsApp.", 0],
    ["Can you draft an email to my boss about the meeting?", 0],
    ["Send a WhatsApp message to Jane saying I'll be late.", 0],
    ["Compose a text to Mike reminding him about the party.", 0],
    ["Email my friend Sarah about the upcoming trip.", 0],
    ["Send a message to my coworker about the project deadline.", 0],
    ["Please send a text to my friend using the AI assistant.", 0],
    ["Let me email my colleague using the AI personal assistant.", 0],
    ["Text my family using the AI chatbot.", 0],
    ["Send a message to my friend using the virtual assistant.", 0],
    ["Compose an email to my client with the help of the AI bot.", 0],
    ["Read my text messages using the AI-powered assistant.", 0],
    ["Draft an email to my team using the virtual assistant.", 0],
    ["Ask the AI chatbot to send a text to my coworker.", 0],
    ["Send a message to my friend via the AI messaging assistant.", 0],
    ["What's around me? Describe my surroundings.", 1],
    ["Describe the scene in front of me.", 1],
    ["Identify the objects around me.", 1],
    ["Describe the people around me.", 1],
    ["What's the weather like right now?", 1],
    ["Read the labels on the nearby items.", 1],
    ["Tell me about the landmarks around me.", 1],
    ["Can you read the street signs nearby?", 1],
    ["Are there any obstacles in my path?", 1],
    ["Please describe the environment around me using the AI assistant.", 1],
    ["Ask the virtual assistant to identify objects around me.", 1],
    ["Describe the scene using the AI-powered chatbot.", 1],
    ["What's the weather like right now according to the AI bot?", 1],
    ["Read the labels on the nearby items with the help of the virtual assistant.", 1],
    ["Describe the people around me using the AI chatbot.", 1],
    ["Ask the virtual assistant to read the street signs nearby.", 1],
    ["Check for obstacles in my path using the AI personal assistant.", 1],
    ["Is there someone I know near me?", 2],
    ["Is my friend Sarah nearby?", 2],
    ["Can you check if my friend Tom is close?", 2],
    ["Find my friend's location on the map.", 2],
    ["Alert me when my friend arrives.", 2],
    ["Notify me if my friend is within 100 meters.", 2],
    ["Help me locate my friend in a crowded place.", 2],
    ["Can you tell me if my friend is on the same street?", 2],
    ["Detect if my friend is in the same room.", 2],
    ["Is my friend John nearby according to the AI assistant?", 2],
    ["Ask the virtual assistant to check if my friend is close.", 2],
    ["Find my friend's location on the map with the help of the AI bot.", 2],
    ["Alert me when my friend arrives using the AI-powered personal assistant.", 2],
    ["Notify me if my friend is within 100 meters using the virtual assistant.", 2],
    ["Help me locate my friend in a crowded place with the face detector.", 2],
    ["Can you tell me if my friend is on the same street using the AI assistant?", 2],
    ["Detect if my friend is in the same room using the virtual assistant.", 2],
    ["Is my friend Jane nearby according to the face detector?", 2],
    ["Ask the virtual assistant to check if my friend Tom is close.", 2],
    ["Find my friend's location on the map with the help of the AI personal assistant.", 2],
    ["Alert me when my friend Sarah arrives using the face detector.", 2],
    ["Notify me if my friend John is within 100 meters using the virtual assistant.", 2],
    ["Help me locate my friend in a crowded place using the AI face.", 2],
    ["Can you tell me if my friend is on the same street according to the face recognizer?", 2],
    ["Detect if my friend is in the same room using the face detector.", 2],
    ["Call my friend", 3],
    ["Ring my neigbor", 3],
    ["Send a Call to my mom", 3],
    ["Call my friend and tell him to have a safe travel", 3],
    ["Call my mom to pick me up", 3],
    ["Call my sister and wish her a happy birthday.", 3],
    ["I need to call my friend Chris to discuss our travel plans.", 3],
    ["Can you help me make a call to my doctor's office to schedule an appointment?", 3],
    ["Call John and ask him to bring me some groceries on his way back.", 3],
    ["I need to call the pharmacy to refill my prescription.", 3],
    ["Can you make a call to my friend", 3],
    ["Call Mom and let her know I arrived safely at my destination.", 3],
    ["I need to call the bank to check my account balance.", 3],
    ["Call my colleague and ask for an update on the project.", 3],
    ["Can you help me call the customer service to inquire about my order?", 3],
    ["Call my friend Lisa and ask her to read a news article to me.", 4],
    ["Read the latest email from my boss and summarize it for me.", 4],
    ["Can you read the label that is in front of me", 4],
    ["What does that say", 4],
    ["Tell me what the text in front of me says", 4],
    ["Can you read the text message from my friend Sarah and reply with a voice message.", 4],
    ["I need to have the AI assistant read the directions to the nearest coffee shop.", 4],
    ["Read the contents of this document to me and highlight important points.", 4],
    ["Can you read the labels on this package and tell me what's inside?", 4],
    ["Read the captions of the images in this photo album using the AI assistant.", 4],
    ["I need to have the AI-powered chatbot read the ingredients list on this food package.", 4],
    ["Read the news headlines and summarize the top stories for me.", 4]
]

random.shuffle(data)

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
