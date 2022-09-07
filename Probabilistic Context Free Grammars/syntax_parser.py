import json
import numpy as np
from collections import defaultdict

class RuleWriter(object):
    """
    This class is for writing rules in a format
    the judging software can read
    Usage might look like this:

    rule_writer = RuleWriter()
    for lhs, rhs, prob in out_rules:
        rule_writer.add_rule(lhs, rhs, prob)
    rule_writer.write_rules()

    """
    def __init__(self):
        self.rules = []

    def add_rule(self, lhs, rhs, prob):
        """Add a rule to the list of rules
        Does some checking to make sure you are using the correct format.

        Args:
            lhs (str): The left hand side of the rule as a string
            rhs (Iterable(str)): The right hand side of the rule.
                Accepts an iterable (such as a list or tuple) of strings.
            prob (float): The conditional probability of the rule.
        """
        assert isinstance(lhs, str)
        assert isinstance(rhs, list) or isinstance(rhs, tuple)
        assert not isinstance(rhs, str)
        nrhs = []
        for cl in rhs:
            assert isinstance(cl, str)
            nrhs.append(cl)
        assert isinstance(prob, float)

        self.rules.append((lhs, nrhs, prob))


    def write_rules(self, filename="q1.json"):
        """Write the rules to an output file.

        Args:
            filename (str, optional): Where to output the rules. Defaults to "q1.json".
        """
        json.dump(self.rules, open(filename, "w"))

# load the parsed sentences
psents = json.load(open("parsed_sents_list.json", "r"))
#psents = [['A', ['B', ['C', 'blue']], ['B', 'cat']]] # test case

# print a few parsed sentences
# NOTE: you can remove this if you like

# TODO: estimate the conditional probabilities of the rules in the grammar

from collections import defaultdict

def list_to_str(l):
    s="[\""
    for i in l:
        s+=(i+"\",\"")
    return s[:-2]+"]"

pcfg_counts = defaultdict(lambda: defaultdict(lambda: 0))

def processing(node):
    childs=[]
    for child in node[1:]:
        if type(child)==list:
            childs.append(child[0])
            processing(child)
        else:
            if child=="\"":
                pcfg_counts[node[0]]["[\"\\\"\"]"] += 1
            else:
                pcfg_counts[node[0]]["[\""+child+"\"]"] += 1
            return
    pcfg_counts[node[0]][list_to_str(childs)]+=1

for sent in psents:
    processing(sent)

# TODO: write the rules to the correct output file using the write_rules method

import ast
rw=RuleWriter()

for k,v in pcfg_counts.items():
    k_total=sum([v for k,v in v.items()])
    for k2, v in v.items():
        prob=v/k_total
        rw.add_rule(k,ast.literal_eval(k2),prob)

rw.write_rules()