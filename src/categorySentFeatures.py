import svm
import parameters
import data

import nltk
import numpy

'compute features for the multiclass problem of detecting the sentiment of a category in a sentence'
class CategorySentFeatures:

    'Constructor: compute some basic statistics (such as words in training data), to be able to calculate the features individually per sentence later on'
    'category: The category for which this feature extractor should extract features'
    def __init__(self, train, commonFeatures, category):
        # we use word features and other features that are also used by other feature extractors, so we don't implement the same code here
        self.comFeatures = commonFeatures

        self.category = category

        ### w2v centroid feature: compute centroid of each sentiment
        [self.w2vVocab, self.w2vVocabSize] = self.comFeatures.getw2vModel()
        numCatSentw2v = [0]*4
        self.centroids = [[numpy.zeros(parameters.w2vVecSize)] for i in range(4)]

        # compute centroids
        for sentence in train['sentences']:
            filteredPost = sentence['tokens']
            #filteredPosts = [w.lower() for w in sentence['tokens'] if w.lower() not in self.stopwords]

            # extract sentiments appearing in sentence
            sentiments = set()
            for currCat in sentence['categories']:
                if 'polarity' in sentence['categories'][currCat]:
                    sentiments.add(sentence['categories'][currCat]['polarity'])

            for j in sentiments:
                for w in filteredPost:
                    if w in self.w2vVocab:
                        self.centroids[j-1] += self.w2vVocab[w][0]
                        numCatSentw2v[j-1] += 1

        for j in range(4):
            self.centroids[j] /= numCatSentw2v[j]
            # normalize centroids
            self.centroids[j] /= numpy.linalg.norm(self.centroids[j])

    def getCategory(self):
        return self.category

    'get features for a single sample'
    # index: id of sentence in dataset, since they have to be consecutive for SVM HMM. (ignored in SVM Multiclass)
    # returns a string which matches the format constraints of SVM Multiclass
    def getFeatures(self, sentence, index):

        if self.category not in sentence['categories'] or 'polarity' not in sentence['categories'][self.category]: # ignore sentences, that don't contain the considered sentiment
            return ''

        # feature array, which we will return
        # is a array of tuples, each tuple represent an entry in a sparse vector
        features = []

        # current offset in the feature vector, thus the size of the feature vector before considering the current feature
        offset = 0

        # compute unigram features for sentence
        [unigramFeatures, offset] = self.comFeatures.getUnigramFeatures(sentence, offset)
        features += unigramFeatures

        # compute bigram features for sentence
        #[bigramFeatures, offset] = self.comFeatures.getBigramFeatures(sentence, offset)
        #features += bigramFeatures

        # compute w2vCategory features for sentence
        [w2vCategoryFeatures, offset] = self.comFeatures.getW2VCategoryFeatures(sentence, offset)
        features += w2vCategoryFeatures

        [w2vCatSentFeatures, offset] = self.getW2VCatSentFeatures(sentence, offset)
        features += w2vCatSentFeatures

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

        if data.useGlove:
            [SentenceVectorFeatures, offset] = self.comFeatures.getSentenceVectorFeatures(sentence, offset)
            features += SentenceVectorFeatures

        # for SVM hmm, the feature indices must be in increasing order
        features.sort(key=lambda tup: tup[0]) # sort by first element of tuples

        return str(sentence['categories'][self.category]['polarity']) + svm.sparseVectorToString(features) + '\n'

    'get a set of features which measures the similarity from a sentence to the centroids of all category sentiments'
    def getW2VCatSentFeatures(self, sentence, offset):
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