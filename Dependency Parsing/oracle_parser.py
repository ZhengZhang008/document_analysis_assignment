import json
from dependency_parser import DepContext, DepParser


def normalise_word(word):
    """Normalise a word for feature extraction

    Args:
        word (str): input word

    Returns:
        str: normalised output word
    """
    return word.strip().lower()

def extract_features(ctx):
    """Extract some simple features from a DepContext object
    the features represent the current state.
    They would be useful for training a classifier. Note that training a
    classifier is not required for this question.

    Args:
        ctx (DepContext): The current dependency parser context

    Returns:
        list(str): a list of terms which represent the current context
    """
    #NOTE: You should not modify this function

    # get the top of the stack and the next word in the buffer
    stack_id = ctx.stack_top()
    if ctx.is_buffer_empty():
        buff_id = 0
    else:
        buff_id = ctx.next_buffer_word_id()

    # convert the ids to word strings
    stack_word = ctx.word_id_to_str(stack_id)
    buff_word = ctx.word_id_to_str(buff_id)

    # build a feature list from this context
    feats = [
                normalise_word(stack_word)+'_stack', 
                normalise_word(buff_word)+'_buffer',
            ]

    return feats

def parse_gold(dp, gold):
    """Generates a list of actions that generates a ground truth (gold) parse
    also returns a list of features extracted using the extract_features function

    Args:
        dp (DepParser): An initialised dependency parser object
        gold (list(tuple(int,int))): A ground truth parse for the sentence used to initialise dp

    Returns:
        list(str), list(list(str)): first return value is the list of actions 
        performed these are one of "left", "right", "reduce", "shift" ,
        the second return value is the list of features (called states in the code here).
        The list of actions and list of features should be in the order that the actions were performed
    """
    actions = []
    states = []

    # TODO: implement the parse gold function by modifying 
    # Algorithm 1 from this paper: https://aclanthology.org/C12-1059.pdf 
    # so that it works for unlabelled dependency parsing.
    # if needed you can add helper methods within this file (oracle_parser.py)
    while(dp.ctx.is_buffer_empty()==False):
        left_ids = (dp.ctx.next_buffer_word_id(), dp.ctx.stack_top())
        right_ids = (dp.ctx.stack_top(), dp.ctx.next_buffer_word_id())
        right = (dp.ctx.word_id_to_str(right_ids[0]), dp.ctx.word_id_to_str(right_ids[1]))
        states.append([str(right[0]).lower() + "_stack", str(right[1]).lower() + "_buffer"])
        if left_ids in gold:
            if dp.left_arc():
                actions.append("left")
                continue
        elif right_ids in gold:
            if dp.right_arc():
                actions.append("right")
                continue
        else:
            performed=False
            tmp=[dp.ctx.pop_stack()]
            while(not dp.ctx.is_stack_empty()):
                wid=dp.ctx.pop_stack()
                tmp.append(wid)
                left_ids = (dp.ctx.next_buffer_word_id(), wid)
                right_ids = (wid, dp.ctx.next_buffer_word_id())
                if left_ids in gold or right_ids in gold:
                    while(len(tmp)!=0):
                        dp.ctx.push_stack(tmp.pop())
                    performed=dp.reduce()
                    break
            while (len(tmp) != 0):
                dp.ctx.push_stack(tmp.pop())
            if performed:
                actions.append("reduce")
                continue
            else:
                dp.shift()
                actions.append("shift")
    return actions, states


def run_small_test():
    """Runs a small test to help you check your algorithm is outputting in the correct format
    """
    rules = [('[ROOT]', 'A'),
            ('A', 'B'),
            ('B', 'C'),
            ('C', 'B')]

    test_sentences = ['AB']
    ground_truth = [
        [(0, 1), (1, 2)]
    ]

    # get the actions and features
    for sent,gold in zip(test_sentences, ground_truth):
        dp = DepParser(rules, list(sent))
        actions, states = parse_gold(dp, gold)
        print("Action:", actions)
        print("Features:", states)
        print("")

        # the expected output is:
        # Action: ['right', 'right']
        # Features: [['[root]_stack', 'a_buffer'], ['a_stack', 'b_buffer']]
        assert actions == ['right', 'right']
        assert states == [['[root]_stack', 'a_buffer'], ['a_stack', 'b_buffer']]

def run_large_test():
    """Runs a large test using a real set of rules. And parses of english sentences.
    The solutions are not provided as this is primarily for marking purposes.
    """

    # read in the rule set
    rules = json.load(open("rule_set.json", "r"))
    rules = [(r[0], r[1]) for r in rules["rules"]]

    # read in the sentences and their ground truth parses
    sents = json.load(open("gold_parse.json", "r"))
    test_sentences = sents["seqs"]
    test_gold = sents["parse"]

    # get the actions and states for all sentences
    good_count = 0
    fail_count = 0
    for sent,parse in zip(test_sentences, test_gold):
        # remove the label from the ground truth parse then run the parser
        parse = [(p[0], p[1]) for p in parse]
        dp = DepParser(rules, sent[1:])
        actions, states = parse_gold(dp, parse)

        sdeps = sorted(dp.ctx.get_all_dependencies())
        sparse = sorted(parse)
        if str(sdeps) != str(sparse):
            fail_count += 1
            if fail_count <= 3:
                print("Fail case %d" % fail_count)
                print(sent)
                print("Actions:", actions)
                print("Features:", states[:3], "....")
                print("Parser Dependencies:", end="")
                for i,j in sdeps:
                    print((dp.ctx.word_id_to_str(i), dp.ctx.word_id_to_str(j)), end="")
                print("")
                print("Ground Truth Dependencies:", end="")
                for i,j in sparse:
                    print((dp.ctx.word_id_to_str(i), dp.ctx.word_id_to_str(j)), end="")
                print("")
                print("--------")
        else:
            good_count += 1
            if good_count <= 3:
                print("Success case %d" % good_count)
                print("Sentence:", " ".join(sent))
                print("Actions:", actions)
                print("Features:", states[:3], "....")
                print("--------")

    print("Total fails: %d, Total success: %d" % (fail_count, good_count))



def main():
    # run a small test to check your model is working
    run_small_test()

    # run a much larger test
    run_large_test()

if __name__ == "__main__":
    main()