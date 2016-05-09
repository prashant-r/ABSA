import svm
import parameters
import data

import nltk
import numpy

'compute features for classifying the sentiment of an aspect in a sentence'
class AspectSentFeatures:

    'Constructor: compute some basic statistics (such as words in training data), to be able to calculate the features individually per sentence later on'
    def __init__(self, train, commonFeatures):
        # we use word features and other features that are also used by other feature extractors, so we don't implement the same code here
        self.comFeatures = commonFeatures

        ### w2v centroid feature: compute centroid of each sentiment
        [self.w2vVocab, self.w2vVocabSize] = self.comFeatures.getw2vModel()
        numAspw2v = [0]*4
        self.centroids = [[numpy.zeros(parameters.w2vVecSize)] for i in range(4)]

        # compute centroids
        for sentence in train['sentences']:
            filteredPost = sentence['tokens']
            #filteredPosts = [w.lower() for w in sentence['tokens'] if w.lower() not in self.stopwords]

            # extract sentiments appearing in sentence
            sentiments = set([j for j in sentence['aspects'] if j != 0])

            for j in sentiments:
                for w in filteredPost:
                    if w in self.w2vVocab:
                        self.centroids[j-1] += self.w2vVocab[w][0]
                        numAspw2v[j-1] += 1

        for j in range(4):
            self.centroids[j] /= numAspw2v[j]
            # normalize centroids
            self.centroids[j] /= numpy.linalg.norm(self.centroids[j])

    'get features for a single sample'
    # index: id of sentence in dataset, since they have to be consecutive for SVM HMM. (ignored in SVM Multiclass)
    # returns a string which matches the format constraints of SVM Multiclass
    def getFeatures(self, sentence, index):

        # string to return
        ret = ''

        # note that we have to generate a feature vector for each word in the sentence (sentence is treated as a sequence)
        ind = 0
        for word in sentence['tokens']:

            if sentence['aspects'][ind] == 0: # ignore words that are no aspeccts
                ind+=1
                continue

            # feature array, which we will return
            # is a array of tuples, each tuple represent an entry in a sparse vector
            features = []

            # current offset in the feature vector, thus the size of the feature vector before considering the current feature
            offset = 0

            # compute word features for sentence
            [wordFeatures, offset] = self.comFeatures.getWordFeatures(word, offset)
            features += wordFeatures

            # compute bigram features for sentence
            #[bigramFeatures, offset] = self.comFeatures.getBigramFeatures(sentence, offset)
            #features += bigramFeatures

            # compute context features for current word
            [contextFeatures, offset] = self.comFeatures.getContextFeatures(sentence, offset, True, parameters.contextWindow, ind)
            features += contextFeatures
            [contextFeatures, offset] = self.comFeatures.getContextFeatures(sentence, offset, False, parameters.contextWindow, ind)
            features += contextFeatures

            # add POS tag of current word as a feature
            [posFeature, offset] = self.comFeatures.getPoSFeature(sentence, offset, ind)
            features += posFeature

            # add feature, that expresses if there are negations surrounding the word
            [negationFeatures, offset] = self.comFeatures.getNegationFeatures(sentence, offset, parameters.contextWindow, ind)
            features += negationFeatures

            #[w2vAspSentFeatures, offset] = self.getW2VAspSentFeatures(sentence, offset)
            #features += w2vAspSentFeatures

            [capitalizationFeature, offset] = self.comFeatures.getCapitalizationFeature(sentence, offset)
            features += capitalizationFeature

            [elongatedWordFeature, offset] = self.comFeatures.getElongatedWordFeature(sentence, offset)
            features += elongatedWordFeature

            [emoticonFeatures, offset] = self.comFeatures.getEmoticonFeatures(sentence, offset)
            features += emoticonFeatures

            [punctuationFeatures, offset] = self.comFeatures.getPunctuationFeature(sentence, offset)
            features += punctuationFeatures

            [sentimentFeature, offset] = self.comFeatures.getSentimentFeatures(sentence, offset)
            features += sentimentFeature

            [sentiWordFeatures, offset] = self.comFeatures.getSentiwordFeatures(sentence, offset)
            features += sentiWordFeatures

            # TODO more features
            if data.useGlove:
                [wordFeature, offset] = self.comFeatures.getWordVectorFeatures(sentence, offset, ind)
                features += wordFeature

            [nearestAdjSentFeatures, offset] = self.comFeatures.getNearestAdjSentFeatures(sentence, offset, ind)
            features += nearestAdjSentFeatures


            # for SVM hmm, the feature indices must be in increasing order
            features.sort(key=lambda tup: tup[0]) # sort by first element of tuples

            sentiment = sentence['aspects'][ind]
            ret += str(sentiment) + svm.sparseVectorToString(features) + '\n'

            ind += 1

        return ret

    'get a set of features which measures the similarity from a sentence to the centroids of all aspect sentiments'
    def getW2VAspSentFeatures(self, sentence, offset):
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

                for j in range(4):
                    # cosine similarity to positive centroid
                    features.append((offset+j*self.w2vVocabSize+index, numpy.dot(self.centroids[j], wordVec)[0]))

        offset += 4 * self.w2vVocabSize

        return [features, offset]