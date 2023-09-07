import json
import pandas as pd
import numpy as np
#import spacy
from sklearn.linear_model import LogisticRegression

# read in the data
train_data = json.load(open("sents_parsed_train.json", "r"))[:3000]
test_data = json.load(open("sents_parsed_test.json", "r"))
t = [t for t in train_data if t["relation"]["relation"] == "/people/person/nationality"]

def print_example(data, index):
    """Prints a single example from the dataset. Provided only
    as a way of showing how to access various fields in the
    training and testing data.
    Args:
        data (list(dict)): A list of dictionaries containing the examples
        index (int): The index of the example to print out.
    """
    # NOTE: You may Delete this function if you wish, it is only provided as
    #   an example of how to access the data.
    # print the sentence (as a list of tokens)
    print("Tokens:")
    print(" ".join(data[index]["tokens"]))
    print([[i,data[index]["tokens"][i],data[index]["dep_head"][i]]for i in range(len(data[index]["dep"]))])
    print([[data[index]["tokens"][i],data[index]["dep"][i]]for i in range(len(data[index]["dep"]))])
    # print the entities (position in the sentence and type of entity)
    print("Entities:")
    for entity in data[index]["entities"]:
        print("%d %d %s" % (entity["start"], entity["end"], entity["label"]))
    # print the relation in the sentence if this is the training data
    if "relation" in data[index]:
        print("Relation:")
        relation = data[index]["relation"]
        print("%d:%s %s %d:%s" % (relation["a_start"], relation["a"],
            relation["relation"], relation["b_start"], relation["b"]))
    else:
        print("Test examples do not have ground truth relations.")

def write_output_file(relations, filename = "q3.csv"):
    """The list of relations into a csv file for the evaluation script

    Args:
        relations (list(tuple(str, str))): a list of the relations to write
            the first element of the tuple is the PERSON, the second is the
            GeoPolitical Entity
        filename (str, optional): Where to write the output file. Defaults to "q3.csv".
    """
    out = []
    for person, gpe in relations:
        out.append({"PERSON": person, "GPE": gpe})
    df = pd.DataFrame(out)
    df.to_csv(filename, index=False)

# print a single training example
print("Training example:")
print_example(train_data, 0)

print("---------------")
print("Testing example:")
# print a single testing example
# the testing example does not have a ground
# truth relation
print_example(test_data, 2)

#TODO: build a training/validation/testing pipeline for relation extraction
#       then write the list of relations extracted from the *test set* to "q3.csv"
#       using the write_output_file function.
import json
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

train_data = json.load(open("sents_parsed_train.json", "r"))
train_data,val_data=train_test_split(train_data, test_size=0.1,shuffle=True)
#names=[]
deps={}
poses={}
counter = Counter()
for x_raw in train_data:
  entities=pd.json_normalize(x_raw['entities'])
  dep_head=x_raw['dep_head']
  dep=x_raw['dep']
  pos=x_raw['pos']
  lemmas=x_raw['lemma']
  persons=entities[entities["label"]=="PERSON"]['start'].values
  gpes=entities[entities["label"]=="GPE"]['start'].values
  i_persons=entities[entities["label"]=="PERSON"].values[:,:2]
  i_gpes=entities[entities["label"]=="GPE"].values[:,:2]
  if len(np.concatenate([i_persons,i_gpes]))>15:
    continue
#  for i_entity in np.concatenate([i_persons,i_gpes]):
#    for i in range(*i_entity):
#      names.append(lemmas[i])
  for i_entity in np.concatenate([persons,gpes]):
    i_old=i_entity
    i_new=dep_head[i_old]
    while i_new!=i_old:
      deps[dep[i_old]]=1
      poses[pos[i_old]]=1
      counter[lemmas[i_old]]+=1
      i_old=i_new
      i_new=dep_head[i_new]
    deps[dep[i_old]]=1
    poses[pos[i_old]]=1
    counter[lemmas[i_old]]+=1

w2id={}
deps_poses_limit=25
freq_threshold=10
for i,word in enumerate([word for word,count in counter.most_common() if count>freq_threshold]):
  w2id[word]=i+1

d2id={k:i for i,(k,v) in enumerate(deps.items())}
p2id={k:i for i,(k,v) in enumerate(poses.items())}
dep_input_shape=(deps_poses_limit,len(d2id))
pos_input_shape=(deps_poses_limit,len(p2id))
input_size=len(w2id)+1 #one more digit for <UNK> token

import itertools
def create_input(data,X,Y):
  x_raw=data
  dep_head=x_raw['dep_head']
  dep=x_raw['dep']
  pos=x_raw['pos']
  entities=pd.json_normalize(x_raw['entities'])
  i_persons=entities[entities["label"]=="PERSON"].values[:,:2]
  i_gpes=entities[entities["label"]=="GPE"].values[:,:2]
  if len(np.concatenate([i_persons,i_gpes]))>15:
    return
  lemmas=x_raw['lemma']
  GT=x_raw['relation']['a_start'],x_raw['relation']['b_start']
  for pair in itertools.product(i_persons,i_gpes):
    label=(pair[0][0],pair[1][0])
    if label!=GT and np.random.rand()>0.3:
      continue
    x2=[]
    dep_input=np.zeros(dep_input_shape,dtype='uint8')
    pos_input=np.zeros(pos_input_shape,dtype='uint8')
    x=np.zeros(input_size,dtype='uint8')
    for i_entity in pair:
      for i in range(*i_entity):
        if not w2id.get(lemmas[i]):
          x[0]+=1
    i=0
    for p in label:
      i_old=p
      i_new=dep_head[i_old]
      while i_new!=i_old:#for dep path
        if w2id.get(lemmas[i_old]):
          if i<deps_poses_limit:
            if d2id.get(dep[i_old]):
              dep_input[i][d2id[dep[i_old]]]+=1
            if p2id.get(pos[i_old]):
              pos_input[i][p2id[pos[i_old]]]+=1
            i+=1
          x[w2id[lemmas[i_old]]]+=1
        i_old=i_new
        i_new=dep_head[i_new]
      if w2id.get(lemmas[i_old]):#for [root]
        if i<deps_poses_limit:
          if d2id.get(dep[i_old]):
            dep_input[i][d2id[dep[i_old]]]+=1
          if p2id.get(pos[i_old]):
            pos_input[i][p2id[pos[i_old]]]+=1
          i+=1
        x[w2id[lemmas[i_old]]]+=1
    x2.extend(x)
    x2.extend(dep_input.reshape(-1))
    x2.extend(pos_input.reshape(-1))
    X.append(x2)
    if label!=GT:
      Y.append(0)
    else:
      Y.append(1)

X_train=[]
Y_train=[]
X_val=[]
Y_val=[]

for x_raw in train_data:
  create_input(x_raw,X_train,Y_train)
for x_raw in val_data:
  create_input(x_raw,X_val,Y_val)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0,C=100,max_iter=10000)
clf.fit(X_train, Y_train)
s_train=clf.score(X_train, Y_train)
s_val=clf.score(X_val, Y_val)
print("Training set accuracy:",s_train)
print("Validation set accuracy:",s_val)


def create_output(data,Y,clf):
  x_raw=data
  dep_head=x_raw['dep_head']
  dep=x_raw['dep']
  pos=x_raw['pos']
  entities=pd.json_normalize(x_raw['entities'])
  i_persons=entities[entities["label"]=="PERSON"].values[:,:2]
  i_gpes=entities[entities["label"]=="GPE"].values[:,:2]
  if len(np.concatenate([i_persons,i_gpes]))>15:
    return
  lemmas=x_raw['lemma']
  tokens=x_raw['tokens']
  for pair in itertools.product(i_persons,i_gpes):
    label=(pair[0][0],pair[1][0])
    x2=[]
    dep_input=np.zeros(dep_input_shape,dtype='uint8')
    pos_input=np.zeros(pos_input_shape,dtype='uint8')
    x=np.zeros(input_size,dtype='uint8')
    for i_entity in pair:
      for i in range(*i_entity):
        if not w2id.get(lemmas[i]):
          x[0]+=1
    i=0
    for p in label:
      i_old=p
      i_new=dep_head[i_old]
      while i_new!=i_old:#for dep path
        if w2id.get(lemmas[i_old]):
          if i<deps_poses_limit:
            if d2id.get(dep[i_old]):
              dep_input[i][d2id[dep[i_old]]]+=1
            if p2id.get(pos[i_old]):
              pos_input[i][p2id[pos[i_old]]]+=1
            i+=1
          x[w2id[lemmas[i_old]]]+=1
        i_old=i_new
        i_new=dep_head[i_new]
      if w2id.get(lemmas[i_old]):#for [root]
        if i<deps_poses_limit:
          if d2id.get(dep[i_old]):
            dep_input[i][d2id[dep[i_old]]]+=1
          if p2id.get(pos[i_old]):
            pos_input[i][p2id[pos[i_old]]]+=1
          i+=1
        x[w2id[lemmas[i_old]]]+=1
    x2.extend(x)
    x2.extend(dep_input.reshape(-1))
    x2.extend(pos_input.reshape(-1))
    if clf.predict([x2]):
      name=" ".join(tokens[pair[0][0]:pair[0][1]])
      gpe=" ".join(tokens[pair[1][0]:pair[1][1]])
      Y.append([name,gpe])

Y_test=[]
for x_raw in test_data:
  create_output(x_raw,Y_test,clf)
# Example only: write out some relations to the output file
# normally you would use the list of relations output by your model
# as an example we have hard coded some relations from the training set to write to the output file
# TODO: remove this and write out the relations you extracted (obviously don't hard code them)
relations = Y_test
write_output_file(relations)
