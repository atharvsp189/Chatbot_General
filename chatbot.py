import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_bow.h5')

def cleanup_sentences(sent):
    ignore = ['?', '.', ',', '!', '&']
    sent_words = nltk.word_tokenize(sent)
    sent_words = [word for word in sent_words if word not in ignore]
    sent_words = [lemmatizer.lemmatize(word) for word in sent_words]
    return sent_words

def sent_bow(sent):
    sent_words = cleanup_sentences(sent)
    bag = [0] * len(words)
    
    for w in sent_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sent):
    bow = sent_bow(sent)
    res = model.predict(np.array([bow]))[0]
    
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key = lambda x:x[1], reverse=True)
    return_List= []
    for r in results:
        return_List.append({'intent': classes[r[0]], 'probability': str(r[1])})
    
    return return_List

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

while(True):
    message = input("Enter Message: ")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot Response: ", res)