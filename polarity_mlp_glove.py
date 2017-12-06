
# coding: utf-8

# In[1]:


import os
# http://www.cs.cornell.edu/people/pabo/movie-review-data/
# polarity dataset v2.0
# 1000 positive and 1000 negative movie reviews
TEXT_DATA_DIR = 'review_polarity/txt_sentoken/'


# In[2]:


def prepare_texts_labels(TEXT_DATA_DIR):
    texts = []
    labels = []
    # iterate through directories to read each file
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        if not name.startswith('.'):
            path = os.path.join(TEXT_DATA_DIR, name)
            if os.path.isdir(path):
                for fname in sorted(os.listdir(path)):
                    fpath = os.path.join(path, fname)
                    with open(fpath, 'r') as myfile:
                        doc = myfile.read().replace('\n', '')
                        # store each review and label respectively
                        texts.append(doc)
                        if name == 'pos':
                            labels.append(+1)
                        else:
                            labels.append(0)
    return texts, labels


# In[3]:


# Check the numbers of texts and labels
texts, labels = prepare_texts_labels(TEXT_DATA_DIR)
print(len(texts), len(labels))


# In[4]:


# Check the average/maximum/minimum length of reviews in characters
total_avg = sum( map(len, texts) ) / len(texts)
print(total_avg)
print(len(max(texts, key=len)))
print(len(min(texts, key=len)))


# In[5]:


import spacy
import numpy as np


# In[6]:


nlp = spacy.load('en_vectors_glove_md')


# In[7]:


# Tokenize reviews into a doc object
# Check the average/maximum/minimum length of reviews in tokens 
length_list = []

for i, review in enumerate(texts):
    doc = nlp.make_doc(review)
    length_list.append(len(doc))

print(np.mean(length_list), max(length_list), min(length_list))


# In[8]:


# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split
# Split data (and labels) into training and testing subsets
# Stratify the labels for both subsets (i.e., equal numbers of pos/neg reviews)
# Avoid imbalanced class distributions
X_train_data, X_test_data, y_train_label, y_test_label = train_test_split(texts, labels, test_size=0.1, random_state=5, stratify=labels)
print(len(X_train_data), len(y_train_label), y_train_label.count(+1), y_train_label.count(0))
print(len(X_test_data), len(y_test_label), y_test_label.count(+1), y_test_label.count(0))


# In[9]:


# Keep only up to N first words in each review
# This is ad-hoc for this exercise. Other decisions could be made.
MAX_SEQUENCE_LENGTH = 777


# In[10]:


# Functions customized based on https://spacy.io/docs/usage/deep-learning
# Using a simple concatenate representation in this exercise, not treating embedding as a whole 
def get_features(docs, MAX_SEQUENCE_LENGTH, nlp):
    Xs = np.empty((len(list(docs)), MAX_SEQUENCE_LENGTH*nlp.vocab.vectors_length), dtype=np.float64)
    for i, doc in enumerate(docs):
        doc = nlp.make_doc(doc)
        for j, token in enumerate(doc[:MAX_SEQUENCE_LENGTH]):
            if token.has_vector:
                vector = token.vector
            else:
                vector = np.zeros(nlp.vocab.vectors_length)
            for k in range(0, nlp.vocab.vectors_length):
                Xs[i, j*nlp.vocab.vectors_length + k] = vector[k]
    return Xs


# In[11]:


# Get features for training data
X_train = get_features(X_train_data, MAX_SEQUENCE_LENGTH, nlp)
print(X_train.shape, X_train.dtype)


# In[12]:


# Get features for testing data
X_test = get_features(X_test_data, MAX_SEQUENCE_LENGTH, nlp)
print(X_test.shape, X_test.dtype)


# In[13]:


# Converts class labels to numpy arrays
y_train = np.asarray(y_train_label, dtype='float64')
print(y_train.shape, y_train.dtype)
y_test = np.asarray(y_test_label, dtype='float64')
print(y_test.shape, y_test.dtype)


# In[14]:


from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K


# In[15]:


# Create a Sequential model with a linear stack of layers
model = Sequential()


# In[16]:


# Add a hidden layer
# operation in Dense layer: output = activation(dot(input, weight) + bias)
model.add(Dense(units=128, input_dim=MAX_SEQUENCE_LENGTH*nlp.vocab.vectors_length, activation="relu"))


# In[17]:


# Add an output layer
model.add(Dense(units=1, activation='sigmoid'))


# In[18]:


# Configure the learning process before model training
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])


# In[19]:


# Train the model, iterating on the data in batches of 128 samples, 5 epochs
model.fit(X_train, y_train, epochs=5, batch_size=128)


# In[20]:


# Print a summary of your model
print(model.summary())


# In[21]:


# Evaluate the model against the 10% validation data in terms of the loss value & metrics values
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
print(loss_and_metrics)
print("%s: %.2f%%" % (model.metrics_names[1], loss_and_metrics[1]*100))
# Results may vary, because data are split randomly in cell [5]


# In[22]:


# Instantiate a Stratified 10-Folds cross-validator
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
# Avoid constructing folds with imbalanced class distributions
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=13)


# In[23]:


# Repeat the above process for 10 times to get average performance metrics
cvscores = []

X = np.asarray(texts)
Y = np.asarray(labels)

for train, test in kfold.split(X, Y):
    
    X_train = get_features(X[train].tolist(), MAX_SEQUENCE_LENGTH, nlp)    
    y_train = np.asarray(Y[train], dtype='float64')

    X_test = get_features(X[test].tolist(), MAX_SEQUENCE_LENGTH, nlp)    
    y_test = np.asarray(Y[test], dtype='float64')
    
    model = Sequential()
    model.add(Dense(units=128, input_dim=MAX_SEQUENCE_LENGTH*nlp.vocab.vectors_length, activation="relu"))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=128, verbose=0)
    
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


# In[24]:


# Delete session from keras backend to free sources
K.clear_session()

