import xml.etree.ElementTree as ET # note, starting from python 3.3 ET will always try to use the c implementation if possible
#from nltk.tokenize import StanfordTokenizer
import nltk
import os
import numpy as np
import codecs

gloveText = os.path.join('..','utility', 'glove.6B', 'glove.6B.300d.txt')
#gloveText = os.path.join('..','utility', 'glove.840B.300d.txt')
if not os.path.isfile(gloveText):
    gloveText = os.path.join('..','Utility', 'glove.840B.300d.txt')
useGlove = True

#useSenna = True
useSenna = False
sennaPath = os.path.join('..','senna', 'senna')
if not os.path.isfile(sennaPath):
    print('WARNING: SENNA is not installed on your machine. Corresponding features will be omitted.')
    useSenna = False


domains = ['Laptop', 'Restaurant']
sents = ['Positive', 'Negative', 'Conflict', 'Neutral']
'training and test sets that we use in our program'
train = [None]*2
test = [None]*2
# Here is how the datastructure looks like:
# every data set is a dictionary with entries 'sentences' and 'categories':
# 'categories' is a set of all categories appearing in the corresponding training set
# 'sentences is an array of dictionaries, where each dictionary has the entries
#
# 'sentence': the sentence as string
# 'tokens': array of tokens
# 'pos': array of tuples (token, POS-tag)
# 'aspects' is an array of integers with the same length as 'tokens'. If the corresponding token is an aspect, then the integer is unequal 0 and its value determines the polarity
#           1 - positive
#           2 - negative
#           3 - conflict
#           4 - neutral
# 'aspectCategories: array of all categories that appear (each category is a dictionary with 'category' and 'polarity'


'''
lists of positive resp. negative words for sentiment analysis

Use lexicons from here: https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107
'''
positiveWords = set()
negativeWords = set()

'read test and training data from file'
def read():
    global train, test

    trees = [None]*4

    dataPath = os.path.join('..','data')
    trees[0] = ET.parse(os.path.join(dataPath,'laptops_train_purified.xml'))
    trees[1] = ET.parse(os.path.join(dataPath,'restaurants_train_purified.xml'))
    trees[2] = ET.parse(os.path.join(dataPath,'laptops-trial.xml'))
    trees[3] = ET.parse(os.path.join(dataPath,'restaurants-trial.xml'))

    # tokenizer to split sentences into words
    #os.environ["CLASSPATH"] = dataPath = os.path.join('..','stanford-postagger-2015-12-09')
    #tokenizer = StanfordTokenizer()
    tokenizer = nltk.tokenize.TweetTokenizer()

    # convert XML tree into our own datastructure
    assert(len(trees) == 4)
    dataSet = None # currently considered set of sentences
    for i in range(len(trees)):
        if i < 2:
            train[i] = {}
            dataSet = train[i]

            # set of all categories that appear in the set
            dataSet['categories'] = set()
        else:
            test[i-2] = {}
            dataSet = test[i-2]
            dataSet['categories'] = train[i-2]['categories'] # a test set needs to have the same categories as its training set

        # array of all sentences
        dataSet['sentences'] = []

        curr = dataSet['sentences']

        j = 0
        for sentence in trees[i].getroot():
            # add a new empty dictionary for each sentence
            curr.append({})
            curr[j]['categories'] = {}

            assert(sentence[0].tag == 'text')
            # Note, that aspectTerms are not always at position sentence[1]
            for sentenceTag in sentence:
                if sentenceTag.tag == 'text':
                    # add plain sentence
                    curr[j]['sentence'] = sentenceTag.text
                    # add tokenized sentence
                    curr[j]['tokens'] = tokenizer.tokenize(sentenceTag.text)
                    # add POS tags
                    curr[j]['pos'] = nltk.pos_tag(curr[j]['tokens'])
                    if useGlove:
                        curr[j]['wordVec'] = getSentenceVector(curr[j]['tokens']);

                    if 'aspects' not in curr[j]:
                        curr[j]['aspects'] = [0] * len(curr[j]['tokens'])

                if sentenceTag.tag == 'aspectTerms':


                    curr[j]['aspects'] = [0] * len(curr[j]['tokens'])

                    for aspectTerm in sentenceTag:

                        fromInd = int(aspectTerm.attrib['from'])
                        toInd = int(aspectTerm.attrib['to'])

                        # extract aspect from sentence
                        aspect = curr[j]['sentence'][fromInd:toInd]
                        tokAsp = tokenizer.tokenize(aspect)

                        # number of whitespaces (which where deleted by tokenizer) before aspect
                        numWS = curr[j]['sentence'][:fromInd].count(' ')
                        # start index of aspect in tokenized string
                        sa = fromInd - numWS

                        # go over all tokens until we find the token where the aspect starts
                        currLen = 0
                        for currInd in range(len(curr[j]['tokens'])):
                            if currLen == sa:
                                for a in tokAsp:
                                    w = curr[j]['tokens'][currInd]

                                    assert(w == a or w.startswith(a))

                                    if aspectTerm.attrib['polarity'] == 'positive':
                                        curr[j]['aspects'][currInd] = 1
                                    elif aspectTerm.attrib['polarity'] == 'negative':
                                        curr[j]['aspects'][currInd] = 2
                                    elif aspectTerm.attrib['polarity'] == 'conflict':
                                        curr[j]['aspects'][currInd] = 3
                                    else: # aspectTerm.attrib['polarity'] == 'neutral':
                                        curr[j]['aspects'][currInd] = 4

                                    currInd += 1

                                break

                            else:
                                currLen += len(curr[j]['tokens'][currInd])


                if sentenceTag.tag == 'aspectCategories':

                    for aspectCategory in sentenceTag:
                        cat = aspectCategory.attrib['category']
                        curr[j]['categories'][cat] = {}

                        polarity = 0
                        if aspectCategory.attrib['polarity'] == 'positive':
                            polarity = 1
                        elif aspectCategory.attrib['polarity'] == 'negative':
                            polarity = 2
                        elif aspectCategory.attrib['polarity'] == 'conflict':
                            polarity = 3
                        else: # aspectCategory.attrib['polarity'] == 'neutral':
                            polarity = 4
                        curr[j]['categories'][cat]['polarity'] = polarity

                        if i < 2: # only for training sets
                            dataSet['categories'].add(aspectCategory.attrib['category']) # note that dataSet['categories'] is a set, so we only add something, if it does not appear yet

            j += 1

    # read sentiment lexicons into memory
    sentimentPath = os.path.join('..','utility', 'sentiment_lists')

    for j in range(2):
        if j == 0:
            filename = os.path.join(sentimentPath,'positive-words.txt')
            currSet = positiveWords
        else:
            filename = os.path.join(sentimentPath,'negative-words.txt')
            currSet = negativeWords

        with open(filename) as f:
            content = f.readlines()
            for line in content:
                line = line.strip()
                if not line.startswith(';') and len(line) > 0:
                    currSet.add(line)


'print training and test data (at least first n entries)'
def printData(n):
    global train, test

    for i in range(4):
        if i < 2:
            curr = train[i]
        else:
            curr = test[i-2]

        count = 0
        for sentence in curr['sentences']:
            if count == n:
                break
            count +=1
            print(sentence)

'There training files initially contained the test sentences, so we have to purify the training file first'
def purifyTrainFile(trainfile, testfile, outfile):
    trainTree = ET.parse(trainfile)
    testTree = ET.parse(testfile)

    # quadratic runtime, but we only need to run this once
    for sentence in testTree.getroot():
        sentence[0].tag
        # check if current sentence appears in training set
        for ts in trainTree.getroot():
            if sentence[0].text == ts[0].text:
                trainTree.getroot().remove(ts)
                break

    trainTree.write(outfile)


def readGloveData():
    global word_vec_dict

    f = codecs.open(gloveText, "r", "utf-8")
    rawData = f.readlines()
    word_vec_dict = {}
    for line in rawData:
        line = line.strip().split()
        tag = line[0]
        vec = line[1:]
        word_vec_dict[tag] = np.array(vec, dtype=float)

    return word_vec_dict

def getWordVector(word):
    global word_vec_dict
    if word in word_vec_dict:
        return word_vec_dict[word]
    return np.zeros_like(word_vec_dict['hi'])

word_vec_dict = None;

def getSentenceVector(sentence):
    list = []
    for word in sentence:
        list.append(getWordVector(word))
    return list



def getSumVectors(tweetData):
    global word_vec_dict
    numNonZero = 0
    vector = np.zeros_like(word_vec_dict['hi'])

    for word in tweetData:
        vec = getWordVector(word.lower())
        vector = vector + vec
        if vec.sum() != 0:
            numNonZero += 1

    if numNonZero:
        vector = vector / numNonZero

    return vector

word_vec_dict = None;

def prepare_dictionary():
    global word_vec_dict
    word_vec_dict = readGloveData(gloveText)


