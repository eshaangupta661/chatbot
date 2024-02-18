import nltk
import os
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn 
import tensorflow as tf 
import random
import json
import pickle 

#Loading Data
with open("intents.json") as file:
    data = json.load(file)
    
#Initialising empty lists
words = []
labels = []
docs_x = []
docs_y = []

#Looping through our data 
for intent in data['intents']:
    for pattern in intent['patterns']:
        pattern = pattern.lower()
        #Creating a list of words
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent['tag'])
        
    if intent['tag'] not in labels:
      labels.append(intent['tag'])
      
stemmer = LancasterStemmer()
words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []


out_empty = [0 for _ in range(len(labels))]
for x, in __doc__ in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in __doc__]
    for w in wrds:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)
#Converting training data into NumPy arrays
training = np.array(training)
output = np.array(output)

#Saving data to disk
with open("data.pickle","wb") as f:
    pickle.dump((words, labels, training, output),f)
    
tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation = 'softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, nepoch = 200, batch_size = 8, show_metric = True)
model.save("model.tflearn")
    
from flask import Flask, render_template, request
import nltk
import datetime
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
stemmer = LancasterStemmer()
seat_count=50
with open("intents.json") as file:
    data = json.load(file)
with open("data.pickle","rb") as f:
    words, labels, training, output = pickle.load(f)
def bag_of_words (s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem (word. lower()) for word in s_words]
        
    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag [i] = 1
    return np.array(bag)

tf.compat.v1. reset_default_graph()

net = tflearn.input_data(shape = [None, len(training [0])])
net = tflearn.fully_connected (net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected (net, len(output[0]), activation = "softmax")
net = tflearn.regression (net)

model = tflearn. DNN (net)
model.load("model.tflearn")


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get')
def get_bot_response():
    global seat_count
    message= request.args.get('msg')
    if message:
        message
        message.lower()
        results = model.predict([bag_of_words (message, words)])[0]
        result_index = np.argmax(results)
        tag = labels [result_index]
        if results [result_index] > 0.5:
            if tag== "book_table":
                seat_count = 1
                response = "Your table has been booked successfully. Remaining tables: " + str(seat_count)
            elif tag == "available_tables":
                response = "There are " + str(seat_count) + " tables available at the moment."
            elif tag=="menu":
                day= datetime.datetime.now()
                day = day.strftime("A")
                ### Add commented code here