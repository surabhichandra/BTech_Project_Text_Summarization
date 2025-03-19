import os
from tkinter.messagebox import NO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

vectorize = lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])




################


# # Gensim Imports
# from gensim.summarization.summarizer import summarize

# Spacy Imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# NLTK Imports
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Sumy Imports
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Other Imports
from string import punctuation
from heapq import nlargest



from transformers import T5ForConditionalGeneration, T5Tokenizer
# initialize the model architecture and weights
headline_model = T5ForConditionalGeneration.from_pretrained("t5-base")
# initialize the model tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")


def hugging_face_transformer(output_string):
  inputs = tokenizer.encode("summarize: " + output_string, return_tensors="pt", max_length=512, truncation=True)
  outputs = headline_model.generate(
          inputs, 
          max_length=100, 
          min_length=50, 
          length_penalty=2.0, 
          num_beams=8)


  return str(tokenizer.decode(outputs[0])).replace('<pad>','')



def gensim_summarize(text_content, percent):
#     # TextRank Summarization using Gensim Library.
#     # Split is false, gensim return strings joined by "\n". if true, gensim will return list
#     summary = summarize(text_content, ratio=(int(percent) / 100), split=False).replace("\n", " ")

#     # Returning NLTK Summarization Output
#     return summary
      return ''


def spacy_summarize(text_content, percent):
    # Frequency Based Summarization using Spacy.
    # Build a List of Stopwords
    stop_words = list(STOP_WORDS)

    # import punctuations from strings library.
    punctuation_items = punctuation + '\n'

    # Loading en_core_web_sm in Spacy
    nlp = spacy.load('en_core_web_sm')

    # Build an NLP Object
    nlp_object = nlp(text_content)

    # Create the dictionary with key as words and value as number of times word is repeated.
    # Scoring words by its occurrence.
    word_frequencies = {}
    for word in nlp_object:
        if word.text.lower() not in stop_words:
            if word.text.lower() not in punctuation_items:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    # Finding frequency of most occurring word
    max_frequency = max(word_frequencies.values())

    # Divide Number of occurrences of all words by the max_frequency
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    # Save a sentence-tokenized copy of text
    sentence_token = [sentence for sentence in nlp_object.sents]

    # Create the dictionary with key as sentences and value as sum of each important word.
    # Scoring sentences by its words.
    sentence_scores = {}
    for sent in sentence_token:
        sentence = sent.text.split(" ")
        for word in sentence:
            if word.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.lower()]

    # Finding number of sentences and applying percentage on it: since we require to show most X% lines in summary.
    select_length = int(len(sentence_token) * (int(percent) / 100))

    # Using nlargest library to get the top x% weighted sentences.
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)

    # Later joining it to get the final summarized text.
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)

    # Returning NLTK Summarization Output
    return summary


def nltk_summarize(text_content, percent):
    # Frequency Based Summarization using NLTK
    # Store a tokenized copy of text, using NLTK's recommended word tokenizer
    tokens = word_tokenize(text_content)

    # Import the stop words from NLTK toolkit
    stop_words = stopwords.words('english')

    # import punctuations from strings library.
    punctuation_items = punctuation + '\n'

    # Create the dictionary with key as words and value as number of times word is repeated.
    # Scoring words by its occurrence.
    word_frequencies = {}
    for word in tokens:
        if word.lower() not in stop_words:
            if word.lower() not in punctuation_items:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

    # Finding frequency of most occurring word
    max_frequency = max(word_frequencies.values())

    # Divide Number of occurrences of all words by the max_frequency
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    # Save a sentence-tokenized copy of text
    sentence_token = sent_tokenize(text_content)

    # Create the dictionary with key as sentences and value as sum of each important word.
    # Scoring sentences by its words.
    sentence_scores = {}
    for sent in sentence_token:
        sentence = sent.split(" ")
        for word in sentence:
            if word.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.lower()]

    # Finding number of sentences and applying percentage on it: since we require to show most X% lines in summary.
    select_length = int(len(sentence_token) * (int(percent) / 100))

    # Using nlargest library to get the top x% weighted sentences.
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)

    # Later joining it to get the final summarized text.
    final_summary = [word for word in summary]
    summary = ' '.join(final_summary)

    # Returning NLTK Summarization Output
    return summary


def sumy_lsa_summarize(text_content, percent):
    # Latent Semantic Analysis is a unsupervised learning algorithm that can be used for extractive text summarization.
    # Initializing the parser
    parser = PlaintextParser.from_string(text_content, Tokenizer("english"))
    # Initialize the stemmer
    stemmer = Stemmer('english')
    # Initializing the summarizer
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words('english')

    # Finding number of sentences and applying percentage on it: since sumy requires number of lines
    sentence_token = sent_tokenize(text_content)
    select_length = int(len(sentence_token) * (int(percent) / 100))

    # Evaluating and saving the Summary
    summary = ""
    for sentence in summarizer(parser.document, sentences_count=select_length):
        summary += str(sentence)
    # Returning NLTK Summarization Output
    return summary


def sumy_luhn_summarize(text_content, percent):
    # A naive approach based on TF-IDF and looking at the “window size” of non-important words between words of high
    # importance. It also assigns higher weights to sentences occurring near the beginning of a document.
    # Initializing the parser
    parser = PlaintextParser.from_string(text_content, Tokenizer("english"))
    # Initialize the stemmer
    stemmer = Stemmer('english')
    # Initializing the summarizer
    summarizer = LuhnSummarizer(stemmer)
    summarizer.stop_words = get_stop_words('english')

    # Finding number of sentences and applying percentage on it: since sumy requires number of lines
    sentence_token = sent_tokenize(text_content)
    select_length = int(len(sentence_token) * (int(percent) / 100))

    # Evaluating and saving the Summary
    summary = ""
    for sentence in summarizer(parser.document, sentences_count=select_length):
        summary += str(sentence)
    # Returning NLTK Summarization Output
    return summary


def sumy_text_rank_summarize(text_content, percent):
    # TextRank is an unsupervised text summarization technique that uses the intuition behind the PageRank algorithm.
    # Initializing the parser
    parser = PlaintextParser.from_string(text_content, Tokenizer("english"))
    # Initialize the stemmer
    stemmer = Stemmer('english')
    # Initializing the summarizer
    summarizer = TextRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words('english')

    # Finding number of sentences and applying percentage on it: since sumy requires number of lines
    sentence_token = sent_tokenize(text_content)
    select_length = int(len(sentence_token) * (int(percent) / 100))

    # Evaluating and saving the Summary
    summary = ""
    for sentence in summarizer(parser.document, sentences_count=select_length):
        summary += str(sentence)
    # Returning NLTK Summarization Output
    return summary


#import all the required libraries
import numpy as np
import pandas as pd
import pickle
from statistics import mode
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup

from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input,LSTM,Embedding,Dense,Concatenate,Attention

from tensorflow.keras.callbacks import EarlyStopping

import nltk
from nltk import word_tokenize
from nltk.stem import LancasterStemmer
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords


def precision_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def decode_sequence(text_content,en_model):
    # Latent Semantic Analysis is a unsupervised learning algorithm that can be used for extractive text summarization.
    # Initializing the parser
    
    try:
        en_out, en_h, en_c = en_model.predict(text_content)
    except:
        print('model 1')

        # Frequency Based Summarization using NLTK
    # Store a tokenized copy of text, using NLTK's recommended word tokenizer
    tokens = word_tokenize(text_content)

    # Import the stop words from NLTK toolkit
    stop_words = stopwords.words('english')

    # import punctuations from strings library.
    punctuation_items = punctuation + '\n'

    # Create the dictionary with key as words and value as number of times word is repeated.
    # Scoring words by its occurrence.
    word_frequencies = {}
    for word in tokens:
        if word.lower() not in stop_words:
            if word.lower() not in punctuation_items:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

    # Finding frequency of most occurring word
    max_frequency = max(word_frequencies.values())

    # Divide Number of occurrences of all words by the max_frequency
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    # Save a sentence-tokenized copy of text
    sentence_token = sent_tokenize(text_content)

    # Create the dictionary with key as sentences and value as sum of each important word.
    # Scoring sentences by its words.
    sentence_scores = {}
    for sent in sentence_token:
        sentence = sent.split(" ")
        for word in sentence:
            if word.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.lower()]

    # Finding number of sentences and applying percentage on it: since we require to show most X% lines in summary.
    select_length = int(len(sentence_token) * (int(50) / 100))

    # Using nlargest library to get the top x% weighted sentences.
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)

    # Later joining it to get the final summarized text.
    final_summary = [word for word in summary]
    summary = ' '.join(final_summary)

    # Returning NLTK Summarization Output
    return summary


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

def outputsumm(input_text):
	stopWords = set(stopwords.words("english"))
	words = word_tokenize(input_text)
	f = dict()
	for i in words:
		i = i.lower()
		if i in stopWords:
			continue
		if i in f:
			f[i] += 1
		else:
			f[i] = 1
	sentences = sent_tokenize(input_text)
	fsent = dict()
	for j in sentences:
		for x, y in f.items():
			if x in j.lower():
				if j in fsent:
					fsent[j] += y
				else:
					fsent[j] = y
	count = 0
	for k in fsent:
		count += fsent[k]
	average = int(count / len(fsent))
	output_text = ''
	for p in sentences:
		if (p in fsent) and (fsent[p] > (1.2 * average)):
			output_text += " "+p
	return output_text


def custom_trained_model(text_content):
    max_text_len=30
    max_summary_len=8
    # encoder inference
    latent_dim=500
    en_model = None
    #/content/gdrive/MyDrive/Text Summarizer/
    #load the model
    try:
        model = models.load_model(r'algorithm\Text_Summarizer.h5', compile=False)
        
        #construct encoder model from the output of 6 layer i.e.last LSTM layer
        en_outputs,state_h_enc,state_c_enc = model.layers[6].output
        en_states=[state_h_enc,state_c_enc]
        #add input and state from the layer.
        en_model = Model(model.input[0],[en_outputs]+en_states)

        # decoder inference
        #create Input object for hidden and cell state for decoder
        #shape of layer with hidden or latent dimension
        dec_state_input_h = Input(shape=(latent_dim,))
        dec_state_input_c = Input(shape=(latent_dim,))
        dec_hidden_state_input = Input(shape=(63,latent_dim))
        
        # Get the embeddings and input layer from the model
        dec_inputs = model.input[1]
        dec_emb_layer = model.layers[5]
        dec_lstm = model.layers[7]
        dec_embedding= dec_emb_layer(dec_inputs)
        
        #add input and initialize LSTM layer with encoder LSTM states.
        dec_lstm(dec_embedding, initial_state=[dec_state_input_h,dec_state_input_c])
    except:
        pass

    return decode_sequence(text_content,en_model)


