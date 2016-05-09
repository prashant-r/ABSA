from os import path, remove
from subprocess import call
import platform
import sys

import parameters
from categorySentFeatures import CategorySentFeatures


# folder where SVM is stored
SVM_HMM_FOLDER = path.join('..', 'svm_hmm')
SVM_MULTICLASS_FOLDER = path.join('..', 'svm_multiclass')
SVM_LIGHT_FOLDER = path.join('..', 'svm_light')
# names of binary files (without learn or classify extension
SVM_HMM_BINARY = 'svm_hmm_'
SVM_MULTICLASS_BINARY = 'svm_multiclass_'
SVM_LIGHT_BINARY = 'svm_'

# folder to temporary store files
TMP_FOLDER = path.join('..', 'tmp')

# filenames
TRAIN_FILE = path.join(TMP_FOLDER,"training.dat")
TEST_FILE = path.join(TMP_FOLDER,"test.dat")
MODEL_FILE = path.join(TMP_FOLDER,"model.dat")
OUT_FILE = path.join(TMP_FOLDER,"out.dat")

'Transform a sparse feature vector to a string'
def sparseVectorToString(vector):
    ret = ''

    for (index,value) in vector:
        ret += ' ' + str(index) + ':' + str(value)

    return ret


'A classifier which uses the SVM struct implementation to learn and apply a model for a given set of sentences'
class SVM:
    '''
    The parameter classifier defines which classifier should be used:
    0 - SVM Light
    1 - SVM Multiclass
    2 - SVM HMM
    '''
    def __init__(self, classifier):
        self.classifier = classifier
        # set up classifier specific arguments
        self.SVM_FOLDER = ''
        self.SVM_BINARY = ''
        if classifier == 0:
            self.SVM_FOLDER = SVM_LIGHT_FOLDER
            self.SVM_BINARY = SVM_LIGHT_BINARY
        elif classifier == 1:
            self.SVM_FOLDER = SVM_MULTICLASS_FOLDER
            self.SVM_BINARY = SVM_MULTICLASS_BINARY
        else:
            self.SVM_FOLDER = SVM_HMM_FOLDER
            self.SVM_BINARY = SVM_HMM_BINARY

        # determine platform to choose right binaries of svm struct
        if ((platform.system() == 'Linux')):
            self.SVM_FOLDER = path.join(self.SVM_FOLDER, 'linux64')
        elif ((platform.system() == 'Darwin' and platform.processor() == 'i386' )):
            self.SVM_FOLDER = path.join(self.SVM_FOLDER, 'mac64')
        else:
            print('System ' + str(platform.uname())  + ' not supported ')
            sys.exit(1)


    '''
    Train an SVM classifier by using the given feature extractor.
    Note, that the given feature extractor is responsible to deliver feature strings that adhere to the required format by the classifier
    '''
    def train(self, train, featureExtractor):

        numTrainSamples = len(train['sentences'])

        # write training features to file
        trainFile = open(TRAIN_FILE, "w")
        for i in range(numTrainSamples):
            trainFile.write(featureExtractor.getFeatures(train['sentences'][i], i+1))
        trainFile.close()



        if self.classifier == 2:
            # Note, in order to get the right parameter c, we have to multiply it by the number of training sample (see description of svm hmm)
            slackMagTradeOff = parameters.svmC_HMM * numTrainSamples
            call([path.join(self.SVM_FOLDER, self.SVM_BINARY+"learn"), '-c', str(slackMagTradeOff), TRAIN_FILE, MODEL_FILE])
        elif self.classifier == 1:
            call([path.join(self.SVM_FOLDER, self.SVM_BINARY+"learn"), '-c', str(parameters.svmC_MC), TRAIN_FILE, MODEL_FILE])
        else:
            call([path.join(self.SVM_FOLDER, self.SVM_BINARY+"learn"), TRAIN_FILE, MODEL_FILE])

        #remove(TRAIN_FILE)

    'resultKey: the name of the key, which we use to store the predicted results in the test struct'
    def predict(self, test, featureExtractor, resultKey):

        # write test features to test file
        testFile = open(TEST_FILE, "w")
        for i in range(len(test['sentences'])):
            testFile.write(featureExtractor.getFeatures(test['sentences'][i], i+1))
        testFile.close()

        call([path.join(self.SVM_FOLDER, self.SVM_BINARY+"classify"), TEST_FILE, MODEL_FILE, OUT_FILE])

        # use predicted values from out_file to assign stance
        outFile = open(OUT_FILE)

        # for each classifier I have to treat the out file differently
        if self.classifier == 0:
            # handle category predictions
            currCat = featureExtractor.getCategory()

            for sentence in test['sentences']:
                if currCat not in sentence['categories']:
                    sentence['categories'][currCat] = {}

                predicted = float(next(outFile).rstrip())
                sentence['categories'][currCat][resultKey] = 1 if predicted >= 0 else 0

        elif self.classifier == 1: # handle sentiment presictions
            if isinstance(featureExtractor, CategorySentFeatures):
                # handle category sentiment predictions
                currCat = featureExtractor.getCategory()

                for sentence in test['sentences']:
                    if currCat not in sentence['categories'] or 'polarity' not in sentence['categories'][currCat]:
                        continue

                    sentence['categories'][currCat][resultKey] = int(next(outFile).rstrip().split(' ', 1)[0])

            else:
                for sentence in test['sentences']:
                    # predictions for each token in the current sequence (sentence)
                    predictions = []
                    ind = 0
                    for token in sentence['tokens']:
                        if sentence['aspects'][ind] > 0:
                            predictions.append(int(next(outFile).rstrip().split(' ', 1)[0]))
                        else:
                            predictions.append(0)
                        ind += 1

                    sentence[resultKey] = predictions
        else:
            # handle aspect predictions
            for sentence in test['sentences']:
                # predictions for each token in the current sequence (sentence)
                predictions = []
                for token in sentence['tokens']:
                    predictions.append(int(next(outFile).rstrip()) - 1) # minus 1, because SVM struct thus not allow tag IDs 0

                sentence[resultKey] = predictions

        outFile.close()

        #remove(TEST_FILE)
        #remove(MODEL_FILE)
        #remove(OUT_FILE)
