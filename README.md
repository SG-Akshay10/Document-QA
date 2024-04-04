# Document-QA

Document-QA is a comprehensive toolkit designed for implementing question answering on long texts or documents. It utilizes various embedding techniques such as Glove Embedding, Word2Vec, Bag Of Words, and the powerful BERT model to extract accurate answers from extensive textual data. Whether you're working with research papers, articles, or any other type of document, this project provides efficient methods to tackle complex queries and enhance document comprehension.

# Table of Contents
* Introduction
* Techniques
  * Chunking
  * Sentence Tokenization
* Embedding Models
  * Bag of Words and Cosine Similarity
  * Word2Vec Model
  * GloVe Embeddings
  * BERT Model
  * Google Flan T5 Model
* Scoring Module
* Usage
* Results and Inference
* Contributing

# Introduction
The Document Based Question Answering System is designed to provide relevant answers to questions based on the content of a given document. It combines various techniques, including chunking, sentence tokenization, and different embedding models, to extract and rank relevant snippets or sentences from the document.

# Techniques
### Chunking
Chunking is the process of dividing a large text into smaller, more manageable chunks for efficient processing. This technique is particularly useful when dealing with large documents that may exceed the input length limitations of certain models or when parallelizing the processing across multiple resources.

### Sentence Tokenization
Sentence tokenization is the process of splitting a text into individual sentences. This technique is often used as a preprocessing step for various natural language processing tasks, including question answering. In the context of this project, sentence tokenization is employed to break down the document into individual sentences, which can then be processed and ranked based on their relevance to the given question.

### Embedding Models
This project leverages various embedding models to represent the document text and the question in a vector space, enabling the calculation of similarities between them. The following embedding models are utilized:

### Bag of Words and Cosine Similarity
The Bag of Words (BoW) model represents text as a collection of words, disregarding grammar and word order. Cosine similarity is then used to calculate the similarity between the question and the document text, represented as BoW vectors.

### Word2Vec Model
The Word2Vec model is a neural network-based approach that represents words as dense vectors in a high-dimensional space. These word vectors capture semantic and syntactic relationships between words, allowing for efficient similarity calculations.

### GloVe Embeddings
GloVe (Global Vectors for Word Representation) is an unsupervised learning algorithm that generates word embeddings by capturing global word-word co-occurrence statistics from a large corpus.

### BERT Model
BERT (Bidirectional Encoder Representations from Transformers) is a powerful pre-trained language model that can be fine-tuned for various natural language processing tasks, including question answering. This project utilizes the BERT model to extract relevant answers from the document text.

### Flan T5 Model
The Flan T5 model is a large language model based on the T5 architecture, pre-trained on a diverse corpus of text. It is employed in this project for generating natural language answers to the given questions based on the document content.

# Scoring Module
The scoring module is responsible for ranking the extracted snippets or sentences based on their relevance to the given question. Different scoring techniques are employed depending on the embedding model used. For example, cosine similarity is used for the Bag of Words and Word2Vec models, while the BERT model generates start and end scores for potential answer spans.

# Usage
To use this system, follow these steps:

Import the necessary dependencies and models.
Load the document text (e.g., from a PDF file).
Clean and preprocess the text using the provided functions.
Optionally, chunk the text into smaller segments for efficient processing.
Choose the desired embedding model and corresponding question-answering function.
Pass the document text, question, and any additional required parameters to the chosen function.
Analyze the output, which can be a relevant snippet, sentence, or a more comprehensive answer, depending on the model used.


# Results and Inference
The project provides detailed insights into the performance and outputs of the different models and techniques employed. The key observations and inferences are as follows:

* The Bag of Words, Word2Vec, GloVe Embeddings, and Sentence Tokenization models tend to provide relevant snippets or sentences from the given text as their outputs.
* The Chunking model extracts relevant snippets related to specific topics or tasks mentioned in the document.
* The Bag of Words model provides a more literal answer by extracting relevant sentences directly related to the question.
* The Word2Vec and GloVe Embeddings models sometimes provide outputs that are not directly relevant to the given question.
* The Sentence Tokenization model extracts relevant sentences but without additional context or information.
* The Chunking model also extracts relevant sentences, similar to the Bag of Words model.
* The Flan-T5 model provides a more comprehensive answer by combining relevant information from the text, making it potentially the most informative for the given question.

# Contributing
Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.
