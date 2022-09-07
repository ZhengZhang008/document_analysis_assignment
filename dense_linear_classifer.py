import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
# read the data
df = pd.read_csv("data/labelled_movie_reviews.csv")

# shuffle the rows
df = df.sample(frac=1, random_state=123).reset_index(drop=True)

# get the train, val, test splits
train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
Xr = df["text"].tolist()
Yr = df["label"].tolist()
train_end = int(train_frac*len(Xr))
val_end = int((train_frac + val_frac)*len(Xr))
X_train = Xr[0:train_end]
Y_train = Yr[0:train_end]
X_val = Xr[train_end:val_end]
Y_val = Yr[train_end:val_end]
X_test = Xr[val_end:]
Y_test = Yr[val_end:]

data = dict(np.load("data/word_vectors.npz"))
w2v = {w:v for w, v in zip(data["words"], data["vectors"])}

from nltk.tokenize import TreebankWordTokenizer
tokenize = TreebankWordTokenizer().tokenize
word_unknown=np.ones(300)*-1
# convert a document into a vector
def document_to_vector(doc):
    """Takes a string document and turns it into a vector
    by aggregating its word vectors.

    Args:
        doc (str): The document as a string

    Returns:
        np.array: The word vector this will be 300 dimensionals.
    """
    # TODO: tokenize the input document
    doc=tokenize(doc)
    vecs=np.array([w2v.get(word,word_unknown) for word in doc])
    x=(vecs-vecs.mean(axis=1).reshape(len(vecs),1))
    x=x/(1e-10+vecs.std(axis=1).reshape(len(vecs),1))
    #85.46
    vec=x.mean(axis=0)
    # TODO: aggregate the vectors of words in the input document
    return vec

from scipy import stats
# fit a linear model
def fit_model(Xtr, Ytr, C):
    """Given a training dataset and a regularization parameter
        return a linear model fit to this data.

    Args:
        Xtr (list(str)): The input training examples. Each example is a
            document as a string.
        Ytr (list(str)): The list of class labels, each element of the
            list is either 'neg' or 'pos'.
        C (float): Regularization parameter C for LogisticRegression

    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    #TODO: convert each of the training documents into a vector
    X_train=[]
    Y_train=[]
    for x,y in zip(Xtr,Ytr):
        X_train.append(document_to_vector(x))
        Y_train.append(y)
    #TODO: train the logistic regression classifier
    X_train=np.array(X_train)
    X_mean=X_train.mean(axis=0)
    X_std=X_train.std(axis=0)
    X_train=(X_train-X_mean)/X_std
    model = LogisticRegression(max_iter=5000,C=C)
    model.fit(X_train, Y_train)
    return model,X_mean,X_std,X_train,Y_train

# fit a linear model
def test_model(model, Xtst, Ytst,X_mean,X_std):
    """Given a model already fit to the data return the accuracy
        on the provided dataset.

    Args:
        model (LogisticRegression): The previously trained model.
        Xtst (list(str)): The input examples. Each example
            is a document as a string.
        Ytst (list(str)): The input class labels, each element
            of the list is either 'neg' or 'pos'.

    Returns:
        float: The accuracy of the model on the data.
    """
    #TODO: convert each of the testing documents into a vector
    X_test = []
    Y_test = []
    for x, y in zip(Xtst, Ytst):
        X_test.append(document_to_vector(x))
        Y_test.append(y)
    X_test=(X_test-X_mean)/X_std

    #TODO: test the logistic regression classifier and calculate the accuracy
    score=model.score(X_test,Y_test)
    return score

# TODO: search for the best C parameter using the validation set
model,X_mean,X_std,X,Y=fit_model(X_val,Y_val,1)
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,0.15,0.2,0.5,1,5,10] }
gs = GridSearchCV(model, param_grid)
gs.fit(X , Y)
print(gs.best_params_)
C=gs.best_params_['C']

# TODO: fit the model to the concatenated training and validation set
#   test on the test set and print the result
X=X_train+X_val
Y=Y_train+Y_val
model,X_mean,X_std,_,_=fit_model(X,Y,C)
score=test_model(model,X_test,Y_test,X_mean,X_std)
print(score)



