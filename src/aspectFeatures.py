import svm
import parameters
import data

'compute features for classifying a word as aspect in a sentence'
class AspectFeatures:

    'Constructor: compute some basic statistics (such as words in training data), to be able to calculate the features individually per sentence later on'
    def __init__(self, train, commonFeatures):
        # we use word features and other features that are also used by other feature extractors, so we don't implement the same code here
        self.train = train

        self.comFeatures = commonFeatures

        # TODO features that are unique for this extractor

    'get features for a single sample'
    # index: id of sentence in dataset, since they have to be consecutive.
    # returns a string which matches the format constraints of SVM HMM
    def getFeatures(self, sentence, index):

        if data.useSenna:
            [srlFeatures, srlOffset] = self.comFeatures.getSennaSRLFeatures(sentence)

        # string to return
        ret = ''

        # note that we have to generate a feature vector for each word in the sentence (sentence is treated as a sequence)
        ind = 0
        for word in sentence['tokens']:

            # feature array, which we will return
            # is a array of tuples, each tuple represent an entry in a sparse vector
            features = []

            # current offset in the feature vector, thus the size of the feature vector before considering the current feature
            offset = 0

            if data.useSenna:
                features += srlFeatures[ind]
                offset += srlOffset

            # compute word features for sentence
            [wordFeatures, offset] = self.comFeatures.getWordFeatures(word, offset)
            features += wordFeatures

            # compute context features for current word
            [contextFeatures, offset] = self.comFeatures.getContextFeatures(sentence, offset, True, parameters.contextWindow, ind)
            features += contextFeatures
            [contextFeatures, offset] = self.comFeatures.getContextFeatures(sentence, offset, False, parameters.contextWindow, ind)
            features += contextFeatures


            # add POS tag of current word as a feature
            [posFeature, offset] = self.comFeatures.getPoSFeature(sentence, offset, ind)
            features += posFeature

            # add word vector of current word as a feature

            if data.useGlove:
                [wordFeature, offset] = self.comFeatures.getWordVectorFeatures(sentence, offset, ind)
                features += wordFeature

            # TODO more features

            # for SVM hmm, the feature indices must be in increasing order
            features.sort(key=lambda tup: tup[0]) # sort by first element of tuples

            isAspect = 2 if sentence['aspects'][ind] > 0 else 1
            ret += str(isAspect) + " qid:" + str(index) + svm.sparseVectorToString(features) + '\n'

            ind += 1

        return ret