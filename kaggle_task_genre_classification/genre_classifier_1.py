
# NOTE: This file contains is a very poor model which looks for manually 
# chosen keywords and if none are found it predicts randomly according
# to the class distribution in the training set
from transformers import AutoTokenizer,BertForSequenceClassification,Trainer,TrainingArguments
import torch
import json
import numpy as np

# load the training data
train_data = json.load(open("genre_train.json", "r"))
X = train_data['X']
Y = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humor, 3 - Crime Fiction
docid = train_data['docid'] # these are the ids of the books which each training example came from

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.encodings["input_ids"])

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
X = tokenizer(X, padding="max_length", truncation=True, max_length=200, return_tensors="pt")


# load the test data
# the test data does not have labels, our model needs to generate these
test_data = json.load(open("genre_test.json", "r"))
Xt = test_data['X']
# This is a very poor model which looks for keywords and if none are found it predicts
# randomly according to the class distribution in the training set
Xt = tokenizer(Xt, padding="max_length", truncation=True, max_length=200, return_tensors="pt")



class TransformerModel(object):
    def __init__(self):
        self.model= BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=4)
        self.trainer = None

    def fit(self, X, Y):
        trainset = Dataset(X, Y)
        training_args = TrainingArguments("test_trainer")
        training_args.per_device_train_batch_size = 32
        self.trainer = Trainer(model=self.model, args=training_args, train_dataset=trainset)
        self.trainer.train()

    def predict(self, Xin):
        testset = Dataset(Xin)
        raw_pred, _, _ = self.trainer.predict(testset)
        Y_test_pred = raw_pred.argmax(axis=1)
        return Y_test_pred



#fit the model on the training data
model = TransformerModel()
model.fit(X, Y)

# predict on the test data
Y_test_pred = model.predict(Xt)
# write out the csv file
# first column is the id, it is the index to the list of test examples
# second column is the predction as an integer
fout = open("out.csv", "w")
fout.write("Id,Y\n")
for i, line in enumerate(Y_test_pred): # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()
