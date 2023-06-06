

import io
import queue
import pyaudio
import time
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import sys
import json
import speech_recognition as sr
import pyttsx3
import random
import string 
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) 


#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only




with open('ai.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey","سلام عليكم")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me","تفضل معلم"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        
# model = Model(r'C:/Users/EBTECH/Downloads/box/py/vosk-model-en-us-0.42-gigaspeech')
# device_info = sd.query_devices(sd.default.device[0], 'input')
# samplerate = int(device_info['default_samplerate'])


# q = queue.Queue()

# def recordCallback(indata, frames, time, status):
#     if status:
#         print(status, file=sys.stderr)
#     q.put(bytes(indata))
    
# # build the model and recognizer objects.
# print("===> Build the model and recognizer objects.  This will take a few minutes.")
# model = Model(r'C:/Users/EBTECH/Downloads/box/py/vosk-model-en-us-0.42-gigaspeech')
# recognizer = KaldiRecognizer(model, samplerate)
# recognizer.SetWords(False)

# def takecommand():
#    while True:
#        try:
    
#          r = sr.Recognizer()
#          t = recognizerResult 


#          try:
#           with sd.RawInputStream(dtype='int16',
#                                   channels=1,

#                                   callback=recordCallback):
          
#             data = q.get(timeout=5)        
#             if recognizer.AcceptWaveform(data):
#                 recognizerResult = recognizer.Result()
#                 # convert the recognizerResult string into a dictionary  
#                 resultDict = json.loads(recognizerResult)
#                 if not resultDict.get("text", "") == "":
#                     print(recognizerResult)
#          except Exception as e:
#            print(str(e))
#            return recognizerResult
           
          
#        except: 
#         continue
#        return t
def takecommand():
    
    r = sr.Recognizer()
    
    with sr.Microphone() as source:
        
        print("3am asma3k")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language ='en-ar')
        print(f"abo ahmad said: {query}\n")
        
    except Exception as e:
        print(e)
        print("Unable to Recognize your voice. معلم صوتك مو واضح ")
        return "None ولا شي"
    
    return query

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
def speak(audio):
    engine.say(audio)
    engine.runAndWait()



def response(abo_ahmad_response):
    robo_response=''
    sent_tokens.append(abo_ahmad_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response
    return robo_response
def t():
   tr = response(abo_ahmad_response)
   print("Nabil: ",end="")
   print(tr)
   sent_tokens.remove(abo_ahmad_response)

   return tr


flag=True
print("Nabil: My name is NAbil. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    abo_ahmad_response =  takecommand() or input()
    
    abo_ahmad_response=abo_ahmad_response.lower()
    if(abo_ahmad_response!='bye'):
        if(abo_ahmad_response=='thanks' or abo_ahmad_response=='thank you' ):
            flag=False
            print("Nabil: You are welcome..")
            speak("Nabil: You are welcome..")

        else:
            if(greeting(abo_ahmad_response)!=None):
                print("Nabil: "+greeting(abo_ahmad_response))
                speak("Nabil: "+greeting(abo_ahmad_response))
            else:
                print("Nabil: ",end="")
                print(response(abo_ahmad_response))
                speak("Nabil: ",)
                hi = response(abo_ahmad_response)
                speak(t)
                print(t)
                sent_tokens.remove(abo_ahmad_response)
    else:
        flag=False
        print("Nabil: Bye! take care..")    
        speak("Nabil: Bye! take care..")    


        

