

import os
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt



! pip install -q kaggle # install kaggle in colab
!rm -r ~/.kaggle 
!mkdir ~/.kaggle 

"""After creating the `~/.kaggle/` directory, we upload the `kaggle.json` file downloaded from the Kaggle user account."""

from google.colab import files
files.upload()

"""And move the `kaggle.json` file into the `~/.kaggle` directory."""

!mv ./kaggle.json ~/.kaggle/ # move the kaggle.json file to the newly created directory
!chmod 600 ~/.kaggle/kaggle.json # change permnission
!ls -l ~/.kaggle/kaggle.json



!kaggle competitions download -c learn-ai-bbc



!mkdir /content/data

!unzip -q /content/learn-ai-bbc.zip -d /content/data/ 
!rm -r /content/learn-ai-bbc.zip



os.listdir('/content/data/')



with open("/content/data/BBC News Train.csv", 'r') as csvfile:
    print(f"CSV header:\n {csvfile.readline()}")
    print(f"First data point:\n {csvfile.readline()}")

with open("/content/data/BBC News Test.csv", 'r') as csvfile:
    print(f"CSV header:\n {csvfile.readline()}")
    print(f"First data point:\n {csvfile.readline()}")


NUM_WORDS = 1000

EMBEDDING_DIM = 16

MAXLEN = 120

PADDING = 'post'

OOV_TOKEN = "<OOV>"

TRAINING_SPLIT = .8



def remove_stopwords(sentence):
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
    
    sentence = sentence.lower()

    word_list = sentence.split()

    words = [w for w in word_list if w not in stopwords]
    
    sentence = " ".join(words)

    return sentence


def parse_data_from_file(filename):
    sentences = []
    labels = []

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader) 

        for row in reader:
            labels.append(row[2])

            sentence = row[1]
            sentence = remove_stopwords(sentence)
            sentences.append(sentence)

    return sentences, labels


sentences, labels = parse_data_from_file("/content/data/BBC News Train.csv")

print(f"Number of sentences in the training dataset: {len(sentences)}\n")
print(f"Number of words in the 1st sentence (after removing stopwords). {len(sentences[0].split())}\n")
print(f"Number of labels in the dataset: {len(labels)}\n")
print(f"First 10 labels: {labels[:10]}")



def train_val_split(sentences, labels, training_split):
    train_size = int(len(sentences) * training_split)

    train_sentences = sentences[0:train_size]
    train_labels = labels[0:train_size]

    validation_sentences = sentences[train_size:]
    validation_labels = labels[train_size:]
 
    return train_sentences, validation_sentences, train_labels, validation_labels


train_sentences, val_sentences, train_labels, val_labels = train_val_split(sentences, labels, TRAINING_SPLIT)

print(f"Number of sentences for training: {len(train_sentences)} \n")
print(f"Number of labels for training: {len(train_labels)}\n")
print(f"Number of sentences for validation: {len(val_sentences)} \n")
print(f"Number of labels for validation: {len(val_labels)}")



def fit_tokenizer(train_sentences, num_words, oov_token):    
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    
    tokenizer.fit_on_texts(train_sentences)
    
    return tokenizer


tokenizer = fit_tokenizer(train_sentences, NUM_WORDS, OOV_TOKEN)

word_index = tokenizer.word_index

print(f"Number of words in the vocabulary: {len(word_index)}\n")


def seq_and_pad(sentences, tokenizer, padding, maxlen):       
    sequences = tokenizer.texts_to_sequences(sentences)
    
    padded_sequences = pad_sequences(sequences, 
                                     maxlen=maxlen, 
                                     padding=padding, 
                                     truncating='post')
    
    return padded_sequences


train_padded_seq = seq_and_pad(train_sentences, tokenizer, PADDING, MAXLEN)
val_padded_seq = seq_and_pad(val_sentences, tokenizer, PADDING, MAXLEN)

print(f"Shape of padded training sequences: {train_padded_seq.shape}\n")
print(f"Shape of padded validation sequences: {val_padded_seq.shape}")


def tokenize_labels(all_labels, split_labels):    
    label_tokenizer = Tokenizer()
    
    label_tokenizer.fit_on_texts(all_labels)

    

    label_seq = label_tokenizer.texts_to_sequences(split_labels)
    
    label_seq_np = np.array(label_seq)-1

    return label_seq_np


train_label_seq = tokenize_labels(labels, train_labels)
val_label_seq = tokenize_labels(labels, val_labels)

print(f"Shape of tokenized labels of the training set: {train_label_seq.shape}\n")
print(f"Shape of tokenized labels of the validation set: {val_label_seq.shape}\n")
print(f"First 5 labels of the training set:\n{train_label_seq[:5]}\n")
print(f"First 5 labels of the validation set:\n{val_label_seq[:5]}\n")



def model(num_words, embedding_dim, maxlen, lstm1_dim, lstm2_dim, num_categories):
    tf.random.set_seed(123)
    
    model = tf.keras.Sequential([ 
        tf.keras.layers.Embedding(num_words, embedding_dim, input_length=maxlen),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm1_dim, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm2_dim)),
        tf.keras.layers.Dense(num_categories, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']) 

    return model


num_unique_categories = np.unique(labels)
print(f'Number of unique categories in the training dataset: {len(num_unique_categories)}')


# set LSTM dimensions
lstm1_dim = 32
lstm2_dim = 16

# create the model
model = model(NUM_WORDS, EMBEDDING_DIM, MAXLEN, lstm1_dim, lstm2_dim, len(num_unique_categories))

print(f'\nModel Summary: {model.summary()}')


# train the model
history = model.fit(train_padded_seq, train_label_seq, epochs=30, validation_data=(val_padded_seq, val_label_seq))



def evaluate_model(history):

    epoch_accuracy = history.history['accuracy']
    epoch_val_accuracy = history.history['val_accuracy']
    epoch_loss = history.history['loss']
    epoch_val_loss = history.history['val_loss']
    
    plt.figure(figsize=(20, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, len(epoch_accuracy)), epoch_accuracy, 'b-', linewidth=2, label='Training Accuracy')
    plt.plot(range(0, len(epoch_val_accuracy)), epoch_val_accuracy, 'r-', linewidth=2, label='Validation Accuracy')
    plt.title('Training & validation accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(range(0, len(epoch_loss)), epoch_loss, 'b-', linewidth=2, label='Training Loss')
    plt.plot(range(0, len(epoch_val_loss)), epoch_val_loss, 'r-', linewidth=2, label='Validation Loss')
    plt.title('Training & validation loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    plt.show()

evaluate_model(history)



def parse_test_data_from_file(filename):
    test_sentences = []

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader) 
        for row in reader:
            sentence = row[1]
            sentence = remove_stopwords(sentence)
            test_sentences.append(sentence)

    return test_sentences

test_sentences = parse_test_data_from_file("/content/data/BBC News Test.csv")

print(f"Number of sentences in the test dataset: {len(test_sentences)}\n")
print(f"Number of words in the 1st sentence (after removing stopwords). {len(test_sentences[0].split())}\n")


test_tokenizer = fit_tokenizer(test_sentences, NUM_WORDS, OOV_TOKEN)

test_word_index = test_tokenizer.word_index

test_padded_seq = seq_and_pad(test_sentences, test_tokenizer, PADDING, MAXLEN)

print(f"Number of words in the test vocabulary: {len(test_word_index)}\n")
print(f"Shape of padded training sequences: {test_padded_seq.shape}\n")


predictions = model.predict(test_padded_seq)

predicted_classes = predictions.argmax(axis=1)
print(f'Predicted classes:\n\n {predicted_classes}')