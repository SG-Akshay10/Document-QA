#!/usr/bin/env python
# coding: utf-8

# # Document Based Question Answering System
# * Base Research Paper: https://arxiv.org/pdf/1805.08092.pdf
# * Other References:
#     * https://ieeexplore.ieee.org/abstract/document/9079274
#     * https://arxiv.org/pdf/1707.07328.pdf
#     * https://arxiv.org/pdf/1810.04805.pdf

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# ### Importing Dependencies

# In[2]:


import fitz
import pdfplumber
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


import re
import nltk
import pprint
import gensim
from gensim import corpora
from gensim.models import Word2Vec  
import gensim.downloader as api  
from sklearn.metrics.pairwise import cosine_similarity
from gensim.parsing.preprocessing import remove_stopwords


# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer


# ### Impoting Document

# In[5]:


my_path = '/home/akshay/SNU_AI_DS/6th_sem/NLP/Project/DocumentQA/Main/Final/1.pdf'


# ### Extracting text from documents

# In[6]:


def pdf_extract(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text() + "\n"
    doc.close()

    # Remove authors' names and specific dataset names
    cleaned_text = re.sub(r'\b[A-Z]+\s[A-Z]\s[A-Z]+(\s-\s[A-Z]\s-\s\d+)\b', '', text)
    
    # Remove section headings
    cleaned_text = re.sub(r'\b\d+\.\s[A-Z]+\b', '', cleaned_text)
    
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text.strip()


# In[7]:


context = pdf_extract(my_path)


# ### Preprocess text

# In[8]:


def clean_sentence(sentence, stopwords=False):
    sentence = sentence.lower().strip() # Convert the sentence to lowercase and remove leading/trailing whitespaces
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence) # Remove any characters that are not alphabets, digits, or whitespaces
    if stopwords:
        sentence = remove_stopwords(sentence) # Optionally remove stopwords from the sentence
    return sentence

def get_cleaned_sentences(tokens, stopwords=False):
    cleaned_sentences = [] # Initialize an empty list to store cleaned sentences
    for row in tokens: # Iterate over each row in the tokens
        cleaned = clean_sentence(row, stopwords) # Clean the sentence using the clean_sentence function
        cleaned_sentences.append(cleaned) # Append the cleaned sentence to the list of cleaned_sentences
    return cleaned_sentences # Return the list of cleaned sentences


# ## Model based on Bag Of Words and Cosine Similarity

# In[9]:


def retrieveAndPrintFAQAnswer(question_embedding, sentence_embeddings, sentences):
    max_sim = -1  # Initialize the maximum similarity score
    index_sim = -1  # Initialize the index of the most similar sentence
    for index, embedding in enumerate(sentence_embeddings):  # Iterate over the sentence embeddings
        # Compute the cosine similarity between the question embedding and the current sentence embedding
        sim = cosine_similarity(embedding, question_embedding)[0][0]
        if sim > max_sim:  # If the current similarity is greater than the maximum similarity found so far
            max_sim = sim  # Update the maximum similarity
            index_sim = index  # Update the index of the most similar sentence
  
    return index_sim  # Return the index of the most similar sentence


def naive_drive(file_name, question):
    pdf_txt = pdf_extract(file_name)  # Extract text from the PDF file
    tokens = nltk.sent_tokenize(pdf_txt)  # Tokenize the text into sentences
    cleaned_sentences = get_cleaned_sentences(tokens, stopwords=True)  # Clean and preprocess the sentences
    cleaned_sentences_with_stopwords = get_cleaned_sentences(tokens, stopwords=False)  # Clean sentences without removing stopwords
    sentences = cleaned_sentences_with_stopwords  # Assign cleaned sentences to 'sentences'
    sentence_words = [[word for word in document.split()] for document in sentences]  # Tokenize each sentence into words

    dictionary = corpora.Dictionary(sentence_words)  # Create a dictionary from the tokenized words
    bow_corpus = [dictionary.doc2bow(text) for text in sentence_words]  # Convert tokenized words into Bag-of-Words representation

    question = clean_sentence(question, stopwords=False)  # Clean the question
    question_embedding = dictionary.doc2bow(question.split())  # Convert the question into a Bag-of-Words representation

    index = retrieveAndPrintFAQAnswer(question_embedding, bow_corpus, sentences)  # Retrieve the index of the most similar sentence
    
    return sentences[index]  


# In[10]:


question = "What are the four different transformer-based embedding models?"
answer = naive_drive(my_path, question)
print(answer)


# ## Word2Vec Model

# In[11]:


v2w_model = None  # Initializing Word2Vec model variable

try:
    v2w_model = gensim.models.KeyedVectors.load('./w2vecmodel.mod')  # Try to load the Word2Vec model from a local file
    print("Word2Vec model successfully loaded")  
except FileNotFoundError:  # Handle the case when the local file is not found
    v2w_model = api.load('word2vec-google-news-300')  # Load the pre-trained "word2vec-google-news-300" model from gensim downloader
    v2w_model.save("./w2vecmodel.mod")  # Save the loaded model to a local file for future use
    print("Word2Vec model saved")  

w2vec_embedding_size = len(v2w_model['pc'])


# In[12]:


def getWordVec(word, model):
    samp = model['pc']
    vec = [0]*len(samp)
    try:
        vec = model[word]
    except:
        vec = [0]*len(samp)
    return (vec)


def getPhraseEmbedding(phrase, embeddingmodel):
    samp = getWordVec('computer', embeddingmodel)
    vec = np.array([0]*len(samp))
    den = 0;
    for word in phrase.split():
        den = den+1
        vec = vec + np.array(getWordVec(word, embeddingmodel))
    return vec.reshape(1, -1)


# In[13]:


def word2vec_drive(file_name, question):
    pdf_txt = pdf_extract(file_name)  # Extract text from the PDF file

    tokens = nltk.sent_tokenize(pdf_txt)  # Tokenize the text into sentences
    cleaned_sentences_with_stopwords = get_cleaned_sentences(tokens, stopwords=False)  # Clean sentences without removing stopwords
    sentences = cleaned_sentences_with_stopwords  # Assign cleaned sentences to 'sentences'
    sentence_words = [[word for word in document.split()] for document in sentences]  # Tokenize each sentence into words

    sent_embeddings = []  # Initialize a list to store embeddings of sentences
    for sent in sentences:  # Iterate over each sentence
        sent_embeddings.append(getPhraseEmbedding(sent, v2w_model))  # Generate the embedding for the sentence and append it to the list

    question_embedding = getPhraseEmbedding(question, v2w_model)  # Generate the embedding for the question
    index = retrieveAndPrintFAQAnswer(question_embedding, sent_embeddings, cleaned_sentences_with_stopwords)  # Retrieve the index of the most similar sentence
    return cleaned_sentences_with_stopwords[index]  # Return the most similar sentence


# In[14]:


answer = word2vec_drive(my_path, question)
print(answer)


# ## Glove Embedding

# In[15]:


glove_model = None
try:
    glove_model = gensim.models.Keyedvectors.load('./glovemodel.mod')
    print("Glove Model Successfully loaded")
except:
    glove_model = api.load('glove-twitter-25')
    glove_model.save("./glovemodel.mod")
    print("Glove Model Saved")

glove_embedding_size = len(glove_model['pc'])


# In[16]:


def glove_drive(file_name, question):
    pdf_txt = pdf_extract(file_name)

    tokens = nltk.sent_tokenize(pdf_txt)
    cleaned_sentences = get_cleaned_sentences(tokens, stopwords=True)
    cleaned_sentences_with_stopwords = get_cleaned_sentences(tokens, stopwords=False)
    sentences = cleaned_sentences_with_stopwords
    sentence_words = [[word for word in document.split()] for document in sentences]

    sent_embeddings = []
    for sent in cleaned_sentences:
        sent_embeddings.append(getPhraseEmbedding(sent, glove_model))

    question_embedding = getPhraseEmbedding(question, glove_model)
    index = retrieveAndPrintFAQAnswer(question_embedding, sent_embeddings, cleaned_sentences_with_stopwords)
    return cleaned_sentences_with_stopwords[index]


# In[17]:


answer = glove_drive(my_path, question)
print(answer)


# # BERT Model

# In[18]:


import torch
from transformers import BertForQuestionAnswering
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


# In[19]:


def answer_question_bert(question, answer_text):

    input_ids = tokenizer.encode(question, answer_text, max_length=512, truncation=True)

    sep_index = input_ids.index(tokenizer.sep_token_id)

    num_seg_a = sep_index + 1

    num_seg_b = len(input_ids) - num_seg_a

    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    assert len(segment_ids) == len(input_ids)

    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids])).values()

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    #print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))
    #print(f'score: {torch.max(start_scores)}')
    score = float(torch.max(start_scores))
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = tokens[answer_start]

    for i in range(answer_start + 1, answer_end + 1):

        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]
        #if tokens[i][0:2] == ' ':
         #   answer += tokens[i][2:]

        #else:
           # answer += ' ' + tokens[i]
    return answer, score, start_scores, end_scores, tokens
    #print('Answer: "' + answer + '"')


# In[20]:


def expand_split_sentences(pdf_text,max_tokens=256):
    tokenized_text = []
    sentences = nltk.sent_tokenize(pdf_text)
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        tokenized_text.extend(tokens)
    
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    for token in tokenized_text:
        current_chunk.append(token)
        current_chunk_tokens += 1
        if current_chunk_tokens >= max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_chunk_tokens = 0
    if current_chunk_tokens > 0:
        chunks.append(' '.join(current_chunk))
    return chunks


# In[21]:


def bert_drive(ftext, question):
    text = ftext
    max_score = 0;
    final_answer = ""
    new_df = expand_split_sentences(text)
    tokens = []
    s_scores = np.array([])
    e_scores = np.array([])
    for new_context in new_df:
    #new_paragrapgh = new_paragrapgh + answer_question(question, answer_text)
        ans, score, start_score, end_score, token = answer_question_bert(question, new_context)
        if score > max_score:
            max_score = score
            s_scores = start_score.detach().numpy().flatten()
            e_scores = end_score.detach().numpy().flatten()
            tokens = token
            final_answer = ans
    return final_answer, s_scores, e_scores, tokens


# In[22]:


def bert_drive(file_name, question):
    # Extract text from PDF
    text = pdf_extract(file_name)
    
    # Initialize variables
    max_score = 0
    final_answer = ""
    new_df = expand_split_sentences(text)
    tokens = []
    s_scores = np.array([])
    e_scores = np.array([])
    
    # Iterate over split sentences and find the best answer
    for new_context in new_df:
        ans, score, start_score, end_score, token = answer_question_bert(question, new_context)
        if score > max_score:
            max_score = score
            s_scores = start_score.detach().numpy().flatten()
            e_scores = end_score.detach().numpy().flatten()
            tokens = token
            final_answer = ans
    
    #return new_df
    return final_answer


# In[23]:


text = pdf_extract(my_path)

answer = bert_drive(my_path, question)
print(answer)


# # Gradio UI

# In[24]:


import gradio as gr


# In[25]:


input_file = gr.File(label="Input File")
question_input = gr.Textbox(label="Question")
output_text = gr.Textbox(label="Answer")


# In[26]:


gr.Interface(fn=bert_drive,inputs=[input_file, question_input],outputs=output_text).launch()

