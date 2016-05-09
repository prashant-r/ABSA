import svm
import parameters
import data

'compute features for the multiclass problem of detecting the category of a sentence (since multiple categories can appear at once, the problem is splitted into multiple binary problems'
class CategoryFeatures:

    'Constructor: compute some basic statistics (such as words in training data), to be able to calculate the features individually per sentence later on'
    'category: The category for which this feature extractor should extract features'
    def __init__(self, train, commonFeatures, category):
        # we use word features and other features that are also used by other feature extractors, so we don't implement the same code here
        self.comFeatures = commonFeatures

        self.category = category

        # TODO features that are unique for this extractor

    def getCategory(self):
        return self.category

    'get features for a single sample'
    # index: id of sentence in dataset, since they have to be consecutive for SVM HMM. (ignored in SVM Light)
    # returns a string which matches the format constraints of SVM Multiclass
    def getFeatures(self, sentence, index):

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

        if data.useGlove:
            [SentenceVectorFeatures, offset] = self.comFeatures.getSentenceVectorFeatures(sentence, offset)
            features += SentenceVectorFeatures

        # for SVM hmm, the feature indices must be in increasing order
        features.sort(key=lambda tup: tup[0]) # sort by first element of tuples

        containsCategory = 1 if self.category in sentence['categories'] else -1

        return str(containsCategory) + svm.sparseVectorToString(features) + '\n'