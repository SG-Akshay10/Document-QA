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


# # Importing Dependencies

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
nltk.download('punkt')
from nltk.corpus import stopwords
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
from langchain.text_splitter import TokenTextSplitter
from PyPDF2 import PdfReader


# # Impoting Document

# # Extracting text from documents

# In[5]:


def extract_pdf_text(pdf_path):
    """
    Function to extract text from a PDF file.

    Parameters:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# # Preprocess text

# In[6]:


def clean_text(text):
    """
    Function to clean the text by removing special characters, stopwords, and other irrelevant information.

    Parameters:
    text (str): Input text to be cleaned.

    Returns:
    str: Cleaned text.
    """
    
    # Remove abnormal characters
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
    # Remove authors' names and specific dataset names
    cleaned_text = re.sub(r'\b[A-Z]+\s[A-Z]\s[A-Z]+(\s-\s[A-Z]\s-\s\d+)\b', '', text)

    # Remove section headings
    cleaned_text = re.sub(r'\b\d+\.\s[A-Z]+\b', '', cleaned_text)

    cleaned_text = re.sub(r" • ", " ", text)  # Replace bullet point with a single space
 
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    

    # Remove stopwords using NLTK
    stop_words = set(stopwords.words('english'))
    cleaned_text = ' '.join(word for word in cleaned_text.split() if word.lower() not in stop_words)

    return cleaned_text


# In[7]:


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


# # Chunking

# In[8]:


def chunk_text(text):
    """
    Function to chunk text into smaller segments for processing.

    Parameters:
    text (str): Input text to be chunked.

    Returns:
    list: List of text chunks.
    """
    splitter = TokenTextSplitter(
        encoding_name="gpt2",
        chunk_size=400,
        chunk_overlap=20
    )
    output = splitter.create_documents([text])
    return output


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


def naive_drive(pdf_text, question): 
    tokens = nltk.sent_tokenize(pdf_text)  # Tokenize the text into sentences
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


# ## Word2Vec Model

# In[10]:


v2w_model = None  # Initializing Word2Vec model variable

try:
    v2w_model = gensim.models.KeyedVectors.load('./w2vecmodel.mod')  # Try to load the Word2Vec model from a local file
    print("Word2Vec model successfully loaded")  
except FileNotFoundError:  # Handle the case when the local file is not found
    v2w_model = api.load('word2vec-google-news-300')  # Load the pre-trained "word2vec-google-news-300" model from gensim downloader
    v2w_model.save("./w2vecmodel.mod")  # Save the loaded model to a local file for future use
    print("Word2Vec model saved")  

w2vec_embedding_size = len(v2w_model['pc'])


# In[11]:


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


# In[12]:


def word2vec_drive(pdf_text, question):

    tokens = nltk.sent_tokenize(pdf_text)  # Tokenize the text into sentences
    cleaned_sentences_with_stopwords = get_cleaned_sentences(tokens, stopwords=False)  # Clean sentences without removing stopwords
    sentences = cleaned_sentences_with_stopwords  # Assign cleaned sentences to 'sentences'
    sentence_words = [[word for word in document.split()] for document in sentences]  # Tokenize each sentence into words

    sent_embeddings = []  # Initialize a list to store embeddings of sentences
    for sent in sentences:  # Iterate over each sentence
        sent_embeddings.append(getPhraseEmbedding(sent, v2w_model))  # Generate the embedding for the sentence and append it to the list

    question_embedding = getPhraseEmbedding(question, v2w_model)  # Generate the embedding for the question
    index = retrieveAndPrintFAQAnswer(question_embedding, sent_embeddings, cleaned_sentences_with_stopwords)  # Retrieve the index of the most similar sentence
    return cleaned_sentences_with_stopwords[index]  # Return the most similar sentence


# ## Glove Embedding

# In[13]:


glove_model = None
try:
    glove_model = gensim.models.Keyedvectors.load('./glovemodel.mod')
    print("Glove Model Successfully loaded")
except:
    glove_model = api.load('glove-twitter-25')
    glove_model.save("./glovemodel.mod")
    print("Glove Model Saved")

glove_embedding_size = len(glove_model['pc'])


# In[14]:


def glove_drive(pdf_text, question):

    tokens = nltk.sent_tokenize(pdf_text)
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


# # BERT

# In[15]:


import torch
from transformers import BertForQuestionAnswering
bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


# In[16]:


def answer_question_bert(question, answer_text):
    """
    Function to answer a given question based on the provided text using BERT model.

    Parameters:
    question (str): Question to be answered.
    answer_text (str): Text containing the answer.

    Returns:
    tuple: Answer, score, start scores, end scores, tokens.
    """

    input_ids = bert_tokenizer.encode(question, answer_text, max_length=512, truncation=True)
    sep_index = input_ids.index(bert_tokenizer.sep_token_id)

    num_seg_a = sep_index + 1

    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    assert len(segment_ids) == len(input_ids)

    start_scores, end_scores = bert_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids])).values()

    all_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)

    #print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))
    #print(f'score: {torch.max(start_scores)}')
    score = float(torch.max(start_scores))
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)
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


# # Sentence Tokenize BERT Model

# In[17]:


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


# In[18]:


def bert_drive_sent(text, question):

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


# # Chunking BERT Model

# In[19]:


# Chunking BERT Model

def bert_drive_chunk(chunk, question):
    """
    Function to drive the BERT model for answering a question.

    Parameters:
    chunk (list): List of text chunks.
    question (str): Question to be answered.

    Returns:
    str: Final answer.
    """
    text_chunks = [doc.page_content for doc in chunk]
    
    # Initialize variables
    max_score = 0
    final_answer = ""
    tokens = []
    s_scores = np.array([])
    e_scores = np.array([])
    
    # Iterate over text chunks
    for chunk in text_chunks:
        ans, score, start_score, end_score, token = answer_question_bert(question, chunk)
        if score > max_score:
            max_score = score
            s_scores = start_score.detach().numpy().flatten()
            e_scores = end_score.detach().numpy().flatten()
            tokens = token
            final_answer = ans
    
    return final_answer


# # Flan T5 Model

# In[20]:


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")


# In[21]:


def flan_t5_drive(chunks, question):
    """
    Function to drive the Flan-T5 model for answering a question.

    Parameters:
    chunks (list): List of text chunks.
    question (str): Question to be answered.

    Returns:
    str: Final answer.
    """
    input_text = "Question: " + question + " Context: "
    max_score = 0
    final_answer = ""
    
    # Iterate over text chunks
    for chunk in chunks:
        input_text_chunk = input_text + chunk.page_content
        input_ids = flan_tokenizer.encode(input_text_chunk, return_tensors="pt")
        output_ids = flan_model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
        output_text = flan_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Implement a scoring mechanism to choose the best answer
        # For example, you could use the length of the answer as a proxy for confidence
        score = len(output_text)
        if score > max_score:
            max_score = score
            final_answer = output_text
    
    return final_answer


# # Model Testing

# In[22]:


file_path = '/home/akshay/Documents/Akshay_Resume.pdf'
question = "Which University did he study?"


# In[23]:


pdf_text = extract_pdf_text(file_path)
cleaned_text = clean_text(pdf_text)
chunks = chunk_text(cleaned_text)


# In[24]:


answer = naive_drive(cleaned_text, question)
print("Bag Of Words : ",answer)


# In[25]:


answer = word2vec_drive(cleaned_text, question)
print("Word2Vec : ",answer)


# In[26]:


answer = glove_drive(cleaned_text, question)
print("Glove Embeddings : ",answer)


# In[27]:


sent = bert_drive_sent(cleaned_text,question)
print("Sentence Tokenization : ",sent)


# In[28]:


chunk = bert_drive_chunk(chunks, question)
print("Chunking : ",chunk)


# In[29]:


flan_t5_answer = flan_t5_drive(chunks, question)
print("Flan-T5 Answer:", flan_t5_answer)


# ## Inference
# 
# #### Similarities:
# 
# * The Bag of Words, Word2Vec, GloVe Embeddings, and Sentence Tokenization models seem to provide relevant snippets or sentences from the given text as their outputs.
# * The Chunking model also extracts a relevant snippet related to image segmentation and license plate isolation.
# 
# #### Differences:
# 
# * The Bag of Words model provides a more literal answer by extracting the sentence "implemented image segmentation algorithms isolate license plates various backgrounds" from the text, which directly answers the question about image segmentation projects.
# * The Word2Vec and GloVe Embeddings models provide the same output, which is not directly related to image segmentation projects. Instead, it talks about utilizing the BERT model for understanding abusive language and improving detection performance, which seems irrelevant to the given question.
# * The Sentence Tokenization model extracts the same sentence as the Bag of Words model, but it doesn't provide any additional context or information.
# * The Chunking model also extracts the relevant sentence about isolating license plates and implementing image segmentation algorithms, similar to the Bag of Words model.
# * The Flan-T5 model provides a more comprehensive answer, combining relevant information from the text. It mentions "Developed Automatic Number Plate Recognition (ANPR) application license plate extraction character recognition" and "Implemented image segmentation algorithms isolate license plates various backgrounds." This answer seems to be the most relevant and informative for the given question.
# 
# 
