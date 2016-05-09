import nltk
from nltk.corpus import sentiwordnet as swn
import gensim # word2vec
import numpy
import re
from os import path, remove, chdir
import subprocess

import parameters
import data
import svm

# filenames
SENNA_IN_FILE = path.join(svm.TMP_FOLDER,"senna_in.txt")
SENNA_OUT_FILE = path.join(svm.TMP_FOLDER,"senna_out.txt")
SENNA_PATH = path.join("..","senna")

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['#', ',', '+'])
'Use this class for features that are used by more than one feature extractor (e.g. a feature extractor for aspects and in the feature for aspect sentiments) to avoid dublicated code'
class Features:

    #stemmer = nltk.PorterStemmer()

    lemmatizer = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')

    'Constructor: compute some basic statistics (such as words in training data), to be able to calculate the features individually per post later on'
    def __init__(self, training):

        self.unigramFeatures = {}  # dictionary: word -> relative index in feature vector
        self.numUnigramFeatures = 0
        self.bigramFeatures = {}  # dictionary: bigram -> relative index in feature vector
        self.numBigramFeatures = 0


        # compute unigram and bigram features
        wordSet = []
        bigramSet = []
        # set of sentences for word2vec
        sentenceSet = []

        for sentence in training['sentences']:
            #lemmatizedPost = [self.lemmatizer.lemmatize(w) for w in post['text']]
            #stemmedPost = [self.stemmer.stem(w) for w in post['text']]
            #bigramSet = bigramSet + list(nltk.bigrams(post['text']))
            bigramSet = bigramSet + list(nltk.bigrams(sentence['tokens']))
            sentenceSet.append(sentence['tokens'])
            filteredPost = sentence['tokens']
            #filteredPost = [w for w in sentence['tokens'] if w.lower() not in self.stopwords]
            for word in filteredPost:
                wordSet.append(word)

        # compute frequency distribution ( we can use it to select only meaningful word features)
        wordSet = nltk.FreqDist(wordSet)
        wordSet = list(wordSet.keys())[:parameters.numUnigrams] # only use the first 100000 most common words as features
        #wordSet = set(wordSet)
        # build feature dictionary with relative indices in sparse vector
        i = 1
        for w in wordSet:
            self.unigramFeatures[w] = i
            i += 1
        self.numUnigramFeatures = i - 1


        # compute frequency distribution ( we can use it to select only meaningful bigram features)
        bigramSet = nltk.FreqDist(bigramSet)
        bigramSet = list(bigramSet.keys())[:parameters.numBigrams] # only use the first 1000 most common words as features
        #bigramSet = set(bigramSet)
        # build feature dictionary with relative indices in sparse vector
        index = 1
        for w in bigramSet:
            self.bigramFeatures[w] = index
            index += 1
        self.numBigramFeatures = index - 1

        ### POS Feature
        # extract all POS tags that might appear
        # Store tags as a (tag -> index) mapping to fasten p computation later on
        self.posDict = {}
        taglist = nltk.data.load('help/tagsets/upenn_tagset.pickle').keys()
        i = 1
        for tag in taglist:
            self.posDict[tag] = i
            i+=1
        self.posDict['#'] = i # dirty hack, since it is the only pos tag, that appears in the tagged sentences, that does not appear in the above tagset

        ### Word2Vec Feature
        # train word2vec model with given sentences
        w2vModel = gensim.models.Word2Vec(sentenceSet, size=parameters.w2vVecSize)
        # build own word2vec vocab, to improve runtime later on)
        self.w2vVocab = {}
        i = 1
        for w in w2vModel.vocab:
            if w.lower() not in self.stopwords: # filter out stop word word vectors
                self.w2vVocab[w] = (w2vModel[w], i)
                i += 1
        self.w2vVocabSize = i-1

        # compute the w2v centroid of all sentences belonging to the same category
        self.categories = training['categories']
        self.numCat = len(training['categories'])
        if self.numCat > 0:

            numCatw2v = [0]*self.numCat
            self.centroids = [[numpy.zeros(parameters.w2vVecSize)] for i in range(self.numCat)]

            # compute centroids
            for sentence in training['sentences']:
                filteredPost = sentence['tokens']
                #filteredPosts = [w.lower() for w in sentence['tokens'] if w.lower() not in self.stopwords]

                i = 0
                for currCat in training['categories']:
                    if currCat in sentence['categories']:
                        for w in filteredPost:
                            if w in self.w2vVocab:
                                self.centroids[i] += self.w2vVocab[w][0]
                                numCatw2v[i] += 1

                    i += 1

            i = 0
            for currCat in training['categories']:
                self.centroids[i] /= numCatw2v[i]
                # normalize centroids
                self.centroids[i] /= numpy.linalg.norm(self.centroids[i])
                i += 1

        ### emoticon feature
        positiveSmileys = """:-)  :) =)  :]  :>  :c) x)  :o) :-D  ;D  :D =D xD XD  :oD""".split()
        self.patternPosEmoticons = "|".join(map(re.escape, positiveSmileys))
        negativeSmileys = """:-(  :( =(  :[  :<  :/ x(  :o(  :C  :\'(  :\'C  ;(""".split()
        self.patternNegEmoticons = "|".join(map(re.escape, negativeSmileys))


    'return gensim w2v model trained on trainings data'
    def getw2vModel(self):
        return [self.w2vVocab, self.w2vVocabSize]

    'get word features of a single word'
    '''
    Word feature means, that it is a BOW model, where we have a set
    of all words appearing in the training set. The set is
    represented as a binary vector, if the vector appears in this set
    the corresponding entry is 1.
    Offset is the relative offset that indices in the sparse feature vector should have.
    The given offset plus the size of the Word Feature vector is returned as new offset.
    '''
    def getWordFeatures(self, word, offset):
        # feature array, which we will return
        # is a array of tuples, each tuple represent an entry in a sparse vector
        features = []

        if word in self.unigramFeatures:
            features.append((offset+self.unigramFeatures[word], 1))

        offset += self.numUnigramFeatures

        return [features, offset]

    '''
    Get unigram features within the context of the word with index "index" in the given
    sentence. The boolean "prev" decides if we consider the previous words (True) or the
    consecutive words. "window" determines how many words before resp. after the current word are
    considered as context.
    '''
    def getContextFeatures(self, sentence, offset, prev, window, index):
        # feature array, which we will return
        # is a array of tuples, each tuple represent an entry in a sparse vector
        features = {} # we use a dictionary at first, to avoid double entriesS

        if prev:
            start = index-window
            end = index -1
        else:
            start = index +1
            end = index + window

        i = start-1
        while (i < end):
            i += 1
            if i < 0:
                continue

            if i >= len(sentence['tokens']):
                break

            word = sentence['tokens'][i]
            if word in self.unigramFeatures:
                features[offset+self.unigramFeatures[word]] = 1

        offset += self.numUnigramFeatures

        return [list(features.items()), offset]

    '''
    Search for a negation within a context window
    '''
    def getNegationFeatures(self, sentence, offset, window, index):
        # feature array, which we will return
        # is a array of tuples, each tuple represent an entry in a sparse vector
        features = []

        start = index-window
        end = index + window

        i = start-1
        while (i < end):
            i += 1
            if i < 0:
                continue

            if i >= len(sentence['tokens']):
                break

            if i == index:
                continue

            word = sentence['tokens'][i]
            if word == 'not' or word.endswith('n\'t'):
                features.append((offset+1, 1))
                break

        offset += 1

        return [features, offset]

    'get unigram features of a single sentence'
    def getUnigramFeatures(self, sentence, offset):
        # feature array, which we will return
        # is a array of tuples, each tuple represent an entry in a sparse vector
        features = []

        #lemmatizedPost = [self.lemmatizer.lemmatize(w) for w in tokenizedPost]
        #stemmedPost = [self.stemmer.stem(w) for w in tokenizedPost]
        #filteredPost = [w.lower() for w in tokenizedPost if w.lower() not in self.stopwords]

        tokenSet = set(sentence['tokens'])
        filteredSet = set(sentence['tokens'])

        # unigram features
        #filteredPost = tokenizedPost
        #filteredPost = [w.lower() for w in set(tokenizedPost)]
        for w in filteredSet:
            if w in self.unigramFeatures:
                features.append((offset+self.unigramFeatures[w], 1))

        offset += self.numUnigramFeatures

        return [features, offset]

    'get unigram features of a single sentence'
    def getBigramFeatures(self, sentence, offset):
        # feature array, which we will return
        # is a array of tuples, each tuple represent an entry in a sparse vector
        features = []

        #lemmatizedPost = [self.lemmatizer.lemmatize(w) for w in tokenizedPost]
        #stemmedPost = [self.stemmer.stem(w) for w in tokenizedPost]
        #filteredPost = [w.lower() for w in tokenizedPost if w.lower() not in self.stopwords]

        # bigram features
        bigrams = set(list(nltk.bigrams(sentence['tokens'])))
        for b in bigrams:
            if b in self.bigramFeatures:
                features.append((offset+self.bigramFeatures[b], 1))

        offset += self.numBigramFeatures

        return [features, offset]

    ''' return the POS tag of a specific word in a sentence as feature '''
    def getPoSFeature(self, sentence, offset, index):
        # feature array, which we will return
        # is a array of tuples, each tuple represent an entry in a sparse vector
        features = []

        features.append((offset+self.posDict[sentence['pos'][index][1]], 1))

        offset += len(self.posDict)

        return [features, offset]

    ## TODO
    ''' return the Glove Word Vector Sum for a sentence'''
    def getWordVectorFeatures(self, sentence, offset, index):
        features = []
        for idx in range(300):
            features.append( (offset + (idx + 1), [sentence['wordVec'][index]][0][idx]))
        offset += 300
        return [features, offset]

    def getSentenceVectorFeatures(self, sentence, offset):
        features = []
        filteredSet = set(sentence['tokens'])
        filteredList = list(filteredSet)
        for idx in range(300):
            toAppend = 0
            for i in range(len(filteredList)):
                if filteredList[i] not in stopwords :
                    toAppend = toAppend + [sentence['wordVec'][i]][0][idx]
            features.append((offset + (idx + 1), toAppend))
        offset += 300
        return [features, offset]



    ## TODO Semantic frame SEMAFOR parser
    '''  return the SEMAFOR features for this sentence'''
    '''  http://www.cs.cmu.edu/~ark/SEMAFOR/'''
    def getSemantcFrameFeatures(self, sentence):
        for word in sentence:
            print(word);

    ## TODO Semantic Role label using SENNA
    ''' http://ml.nec-labs.com/senna/'''
    def getSRL(self, sentence):
        for word in sentence:
            print(word);


    ## TODO Syntactic Parse Tree using SENNA
    ''' http://ml.nec-labs.com/senna/'''
    def getSynParse(self, sentence):
        for word in sentence:
            print(word);

    'get a set of features which measures the similarity from a sentence to the centroids of all categories'
    def getW2VCategoryFeatures(self, sentence, offset):
        # feature array, which we will return
        # is a array of tuples, each tuple represent an entry in a sparse vector
        features = []

        filteredSet = set(sentence['tokens'])

        for w in filteredSet:
            if w in self.w2vVocab:
                val = self.w2vVocab[w]
                wordVec = val[0]
                index = val[1]
                wordVec /= numpy.linalg.norm(wordVec)

                i = 0
                for currCat in self.categories:
                    # cosine similarity to positive centroid
                    features.append((offset+i*self.w2vVocabSize+index, numpy.dot(self.centroids[i], wordVec)[0]))
                    i += 1

        offset += self.numCat * self.w2vVocabSize

        return [features, offset]

    'number of words in the sentence that consists only of capital letters'
    def getCapitalizationFeature(self, sentence, offset):
        # feature array, which we will return
        # is a array of tuples, each tuple represent an entry in a sparse vector
        features = []

        # number of words in sentence that consist only of capital letters
        num = 0

        for w in sentence['tokens']:
            if w.isupper():
                num +=1

        if num > 0:
            features.append((offset+1, num/len(sentence['tokens'])))

        offset += 1

        return [features, offset]

    'number of of elongated words in a sentence'
    def getElongatedWordFeature(self, sentence, offset):
        # feature array, which we will return
        # is a array of tuples, each tuple represent an entry in a sparse vector
        features = []

        # number of elongated words
        num = len(re.findall(r"([a-zA-Z])\1{2,}",sentence['sentence']))

        if num > 0:
                features.append((offset+1, num))

        offset += 1

        return [features, offset]

    'Features for positive resp. negative emoticons'
    def getEmoticonFeatures(self, sentence, offset):
        # feature array, which we will return
        # is a array of tuples, each tuple represent an entry in a sparse vector
        features = []

        # number of positive emoticons
        num = len(re.findall(self.patternPosEmoticons,sentence['sentence']))
        if num > 0:
                features.append((offset+1, 1))

        num = len(re.findall(self.patternNegEmoticons,sentence['sentence']))
        if num > 0:
                features.append((offset+2, 1))

        offset += 2

        return [features, offset]

    'Detect sequences of question and/or exclamation marks'
    def getPunctuationFeature(self, sentence, offset):
        # feature array, which we will return
        # is a array of tuples, each tuple represent an entry in a sparse vector
        features = []

        num = len(re.findall(r"(!!|!\?|\?!|\?\?)[!\?]*", sentence['sentence']))

        if num > 0:
                features.append((offset+1, 1))

        offset += 1

        return [features, offset]

    'Sentiment Feature: look for words in the sentence that appear in a list of positive resp. negative word (consider negation of words).'
    def getSentimentFeatures(self, sentence, offset):
        # feature array, which we will return
        # is a array of tuples, each tuple represent an entry in a sparse vector
        features = []

        lemmatizedSentence = [self.lemmatizer.lemmatize(w) for w in sentence['tokens']]

        numPos = 0
        numNeg = 0

        i = 0
        for w in lemmatizedSentence:
            isPos = False
            isNeg = False

            if w in data.positiveWords:
                isPos = True

            if w in data.negativeWords:
                isNeg = True

            if isPos or isNeg:
                # if negation in last two words appear ('is not good' or '
                isNeg = False

                if i > 0:
                    prev = sentence['tokens'][i-1]
                    if prev == 'not' or prev.endswith('n\'t'):
                        isNeg = True
                if not isNeg and i > 1:
                    prev = sentence['tokens'][i-2]
                    if prev == 'not' or prev.endswith('n\'t'):
                        isNeg = True

                if isNeg:
                    if isPos:
                        numNeg +=1
                    else:
                        numPos +=1
                else:
                    if isPos:
                        numPos +=1
                    else:
                        numNeg +=1

            i += 1

        if numPos > 0:
            features.append((offset+1, numPos))

        if numNeg > 0:
            features.append((offset+2, numNeg))

        offset += 2

        return [features, offset]

    'Adding sentiment scores of synonyms of each word and add this as a feature'
    def getSentiwordFeatures(self, sentence, offset):
        # feature array, which we will return
        # is a array of tuples, each tuple represent an entry in a sparse vector
        features = []

        lemmatizedSentence = [self.lemmatizer.lemmatize(w) for w in sentence['tokens']]

        sumPos = 0
        sumNeg = 0

        i = 0
        for w in lemmatizedSentence:

            ssl = list(swn.senti_synsets(w))

            tmpSumPos = 0
            tmpSumNeg = 0
            tmpCountPos = 0
            tmpCountNeg = 0

            for s in ssl:
                ps = s.pos_score()
                ns = s.neg_score()
                if ps > 0:
                    tmpSumPos += ps
                    tmpCountPos += 1
                if ns > 0:
                    tmpSumNeg += ns
                    tmpCountNeg += 1

            if tmpCountPos > 0:
                tmpSumPos /= tmpCountPos
            if tmpCountNeg > 0:
                tmpSumNeg /= tmpCountNeg

            # is the word sentiment negated?
            isNeg = False
            if i > 0:
                prev = sentence['tokens'][i-1]
                if prev == 'not' or prev.endswith('n\'t'):
                    isNeg = True
            if not isNeg and i > 1:
                prev = sentence['tokens'][i-2]
                if prev == 'not' or prev.endswith('n\'t'):
                    isNeg = True

            if isNeg:
                tmp = tmpSumPos
                tmpSumPos = tmpSumNeg
                tmpSumNeg = tmp

            sumPos += tmpSumPos
            sumNeg += tmpSumNeg

            i += 1

        if sumPos > 0:
            features.append((offset+1, sumPos))

        if sumNeg > 0:
            features.append((offset+2, sumNeg))

        offset += 2

        return [features, offset]

    'Adding the SRL tags of a sentence (thus if it''s a predicate or what argument type it is)'
    def getSennaSRLFeatures(self, sentence):

        chdir(SENNA_PATH)
        '''
        r = ''
        for s in dataset['sentences']:
            r += s['sentence']
            r += '\n'

        print(r)
        '''
        proc = subprocess.Popen(["./senna", '-srl'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output,err = proc.communicate(input=sentence['sentence'].encode('utf-8'))

        output = output.decode().split('\n')
        output = [o.split('\t') for o in output]
        output = [[t.strip() for t in o] for o in output]

        # change back to program directory
        chdir(path.dirname(path.realpath(__file__)))


        # feature array, which we will return
        tokenFeatures = []

        offset = 0

        ind = 0
        for tok in sentence['tokens']:
            features = set() # features for a single token

            if ind < len(output):

                currTok = output[ind]
                if tok not in currTok[0] and ind > 0:
                    currTok = output[ind-1]
                    ind -= 1

                # is current token from SENNA is part of the current token in sentence
                if tok in currTok[0]:

                    arr = currTok

                    for i in range(2,len(arr)):
                        if arr[i].endswith('-V'):
                            features.add((offset+1, 1))
                        elif arr[i].endswith('-A0'):
                            features.add((offset+2, 1))
                        elif arr[i].endswith('-A1'):
                            features.add((offset+3, 1))
                        elif arr[i].endswith('-A2'):
                            features.add((offset+4, 1))
                        elif arr[i].endswith('-A3'):
                            features.add((offset+5, 1))
                        elif arr[i].endswith('-A4'):
                            features.add((offset+6, 1))
                        elif arr[i].endswith('-A5'):
                            features.add((offset+7, 1))
                        else:
                            features.add((offset+8, 1))

            tokenFeatures.append(list(features))
            ind += 1

        return [tokenFeatures, 8]

    'Get the nearest adjective and its sentiment as feature'
    def getNearestAdjSentFeatures(self, sentence, offset, ind):
        # feature array, which we will return
        # is a array of tuples, each tuple represent an entry in a sparse vector
        features = []

        lemmatizedSentence = [self.lemmatizer.lemmatize(w) for w in sentence['tokens']]

        closest = -1
        pos = 0
        neg = 0

        i = 0
        for w in lemmatizedSentence:

            ssl = list(swn.senti_synsets(w))

            currPOS = sentence['pos'][i][1]

            if closest == -1 or abs(ind - i) < abs(ind - closest):

                # if current POS tag is adjective or adverb
                if currPOS.startswith('JJ') or currPOS.startswith('RB'):

                    tmpSumPos = 0
                    tmpSumNeg = 0
                    tmpCountPos = 0
                    tmpCountNeg = 0

                    for s in ssl:
                        ps = s.pos_score()
                        ns = s.neg_score()
                        if ps > 0:
                            tmpSumPos += ps
                            tmpCountPos += 1
                        if ns > 0:
                            tmpSumNeg += ns
                            tmpCountNeg += 1

                    if tmpCountPos > 0:
                        tmpSumPos /= tmpCountPos
                    if tmpCountNeg > 0:
                        tmpSumNeg /= tmpCountNeg

                    pos = tmpSumPos
                    neg = tmpSumNeg
                    closest = i

                    # is the word sentiment negated?
                    isNeg = False
                    if i > 0:
                        prev = sentence['tokens'][i-1]
                        if prev == 'not' or prev.endswith('n\'t'):
                            isNeg = True
                    if not isNeg and i > 1:
                        prev = sentence['tokens'][i-2]
                        if prev == 'not' or prev.endswith('n\'t'):
                            isNeg = True

                    if isNeg:
                        tmp = pos
                        pos = neg
                        neg = tmp

            i += 1

        if closest != -1:

            if pos > 0:
                features.append((offset+1, pos))

            if neg > 0:
                features.append((offset+2, neg))

        offset += 2

        return [features, offset]
