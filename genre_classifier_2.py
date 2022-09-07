# NOTE: This file contains is a very poor model which looks for manually
# chosen keywords and if none are found it predicts randomly according
# to the class distribution in the training set

import json
import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Reshape, Input, LSTM, Bidirectional, GRU, concatenate, Conv1D, MaxPooling1D, Flatten, LayerNormalization
from tensorflow.keras.models import Model
from keras.initializers import Constant
from tensorflow.keras import backend as K
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en.stop_words import STOP_WORDS
import string
from gensim import corpora
from keras.preprocessing import sequence
from nltk.tokenize import TreebankWordTokenizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors

# load the training data
train_data = json.load(open("genre_train.json", "r"))
X = train_data['X']
Y = train_data['Y']  # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humor, 3 - Crime Fiction
docid = train_data['docid']  # these are the ids of the books which each training example came from

# load the test data
# the test data does not have labels, our model needs to generate these
test_data = json.load(open("genre_train.json", "r"))
Xt = test_data['X']


class TfidfBiLstmMLP(object):
    def __init__(self,vocab_size,max_len,shape):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tfidf_in_shape=shape
        self.model=self.model()

    def biLSTM(self):
      modelBiLSTM = Sequential()
      modelBiLSTM.add(Embedding(self.vocab_size + 1, embedding_dim, embeddings_initializer=Constant(embedding_matrix),
                                input_length=self.max_len,trainable=True))
      modelBiLSTM.add(LayerNormalization(epsilon=1e-6))
      modelBiLSTM.add(Bidirectional(LSTM(32, dropout=0.5, return_sequences=True)))
      modelBiLSTM.add(Bidirectional(LSTM(64, dropout=0.5, return_sequences=False)))
      modelBiLSTM.add(Dense(32, activation='tanh'))
      return modelBiLSTM
    def tfidt(self):
      model_tfidf = Sequential()
      model_tfidf.add(
          Conv1D(filters=256, strides=2, kernel_size=3, padding='same', activation='relu', input_shape=self.tfidf_in_shape))
      model_tfidf.add(
          Conv1D(filters=128, strides=2, kernel_size=3, padding='same', activation='relu', input_shape=self.tfidf_in_shape))
      model_tfidf.add(Flatten())
      model_tfidf.add(Dense(units=64, activation='tanh'))
      model_tfidf.add(Dense(units=32, activation='tanh'))
      return model_tfidf
    def model(self):
      bi=self.biLSTM()
      model_tfidf=self.tfidt()
      concat = concatenate([bi.output, model_tfidf.output], axis=-1)
      output = Dense(16, activation='tanh')(concat)
      output = Dense(4, activation='softmax')(output)
      model = Model(inputs=[bi.inputs, model_tfidf.inputs], outputs=output)
      return model

    def fit(self, X, Y):
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[self.custom_f1, 'accuracy'])
        self.model.fit(X,Y, batch_size=256, epochs=100, shuffle=True)

    def custom_f1(self,Y_true, pred):
        TP = K.sum(K.round(K.clip(Y_true * pred, 0, 1)))
        pred_pos = K.sum(K.round(K.clip(pred, 0, 1)))
        precision = TP / (pred_pos + K.epsilon())
        pos = K.sum(K.round(K.clip(Y_true, 0, 1)))
        recall = TP / (pos + K.epsilon())
        f1= 2 * ((precision * recall) / (precision + recall + K.epsilon()))
        return f1

    def predict(self, Xin):
        Y_test_pred=self.model.predict(Xin).argmax(axis=1)
        return Y_test_pred


punctuations = string.punctuation
punctuations = punctuations.replace(',', '')
punctuations = punctuations.replace('.', '')
punctuations = punctuations.replace(',', '')
punctuations = punctuations.replace('?', '')
punctuations = punctuations.replace('!', '')
punctuations=[p for p in punctuations]+["''"]
punctuations=[p for p in punctuations]+["..","...","....",".....","......"]
nlp = spacy.load('en')
tokenizer = Tokenizer(nlp.vocab)
tokenize = TreebankWordTokenizer().tokenize
tokenized_lists = []
for x in X:
    tokens = tokenizer(' '.join(tokenize(x)))
    tokenized_list = nlp(tokens.text)
    tokenized_list = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in
                      tokenized_list]
    tokenized_list = [word for word in tokenized_list if word not in STOP_WORDS and word not in punctuations]
    tokenized_lists.append(tokenized_list)
dic = corpora.Dictionary(tokenized_lists).token2id
X_seqs_with_unknown_token = []
tokenized_lists_with_mask = []
for tokenized_list in tokenized_lists:
    tokenized_list = np.array(tokenized_list)
    tokenized_lists_with_mask.append(' '.join(tokenized_list))
    X_seqs_with_unknown_token.append(np.array([dic[token] + 2 for token in tokenized_list]))
    ids = np.array([dic[token] + 2 for token in tokenized_list])
    if len(ids) >= 10:
        indices = np.random.choice(range(len(ids) - 1), int(np.ceil(len(ids) / (10 if np.random.rand() > 0.68 else 8))),
                                   replace=False)
        ids[indices] = 1
    tokenized_list[indices] = ""
    tokenized_lists_with_mask.append(' '.join(tokenized_list))
    X_seqs_with_unknown_token.append(ids)
X_seqs_with_unknown_token = np.array(X_seqs_with_unknown_token)
max_len = 200
vocab_size = len(dic) + 1

X_seqs_pad = sequence.pad_sequences(X_seqs_with_unknown_token.reshape(-1), max_len)
Y = np.array(Y * 2).reshape(2, -1).T.flatten()

Y = OneHotEncoder().fit_transform(np.append(Y, 0).reshape((-1, 1))).toarray()[:-1]

tokenized_lists_test = []
for x in Xt:
    tokens = tokenizer(' '.join(tokenize(x)))
    tokenized_list = nlp(tokens.text)
    tokenized_list = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in
                      tokenized_list]
    tokenized_list = [word for word in tokenized_list if word not in STOP_WORDS and word not in punctuations]
    tokenized_lists_test.append(tokenized_list)
Xt_seqs = []
tokenized_lists_with_mask_test = []
for tokenized_list in tokenized_lists_test:
    Xt_seqs.append([dic.get(token, -1) + 2 for token in tokenized_list])
    tokenized_lists_with_mask_test.append(' '.join(tokenized_list))

Xt_seqs = np.array(Xt_seqs)
Xt_seqs_pad = sequence.pad_sequences(Xt_seqs.reshape(-1), max_len)

vectorizer = TfidfVectorizer(min_df=2, max_features=10000, strip_accents='unicode', lowercase=True,
                             analyzer='word', token_pattern=r'\w+', use_idf=True,
                             smooth_idf=True, sublinear_tf=True, stop_words='english')
vectorizer.fit(tokenized_lists_with_mask)
X_tfidf = vectorizer.transform(tokenized_lists_with_mask).toarray()
Xt_tfidf = vectorizer.transform(tokenized_lists_with_mask_test).toarray()
X_tfidf = X_tfidf.reshape(X_tfidf.shape[0], X_tfidf.shape[1], 1)
Xt_tfidf = Xt_tfidf.reshape(Xt_tfidf.shape[0], Xt_tfidf.shape[1], 1)

# uncomment to download the pretrained embedding
# wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
# gzip -d GoogleNews-vectors-negative300.bin.gz

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
embedding_dim = 300
embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))
unk_ids = []
for i, word in enumerate(dic.keys()):
    w2v = model.vocab.get(word, 'NAN!')
    if w2v == 'NAN!':
        unk_ids.append(i)
        continue
    embedding_matrix[i + 2] = model[word]
embedding_matrix[1] = embedding_matrix[2:].mean(axis=0)
unk_ids = np.array(unk_ids)
embedding_matrix[unk_ids + 2] = embedding_matrix[2:].mean(axis=0)
del model


# fit the model on the training data
X=[X_seqs_pad, X_tfidf]
Y=Y.astype(int)
model = TfidfBiLstmMLP(vocab_size,max_len,(X_tfidf.shape[1], 1))
model.fit(X, Y)

# predict on the test data
Xt=[Xt_seqs_pad,Xt_tfidf]
Y_test_pred = model.predict(Xt)
# write out the csv file
# first column is the id, it is the index to the list of test examples
# second column is the predction as an integer
fout = open("out.csv", "w")
fout.write("Id,Y\n")
for i, line in enumerate(Y_test_pred):  # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()