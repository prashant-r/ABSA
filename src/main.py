import argparse
import sys
import time

import data
import svm
import aspectFeatures
import aspectSentFeatures
import categoryFeatures
import categorySentFeatures
import features
import evaluation

if __name__ == "__main__":
    '''
    # we gotta have to purify their training files, since the supplied training files do contain the test sentences as well (this has to be done only once)
    import os
    dataPath = os.path.join('..','data')
    data.purifyTrainFile(os.path.join(dataPath,'Laptops_Train.xml'),os.path.join(dataPath,'laptops-trial.xml'),os.path.join(dataPath,'laptops_train_purified.xml'))
    data.purifyTrainFile(os.path.join(dataPath,'Restaurants_Train.xml'),os.path.join(dataPath,'restaurants-trial.xml'),os.path.join(dataPath,'restaurants_train_purified.xml'))
    sys.exit(0)
    '''

    # handle command line arguments
    parser = argparse.ArgumentParser()
    #TODO
    #parser.add_argument('-c', '--class', help='Define classifier (0: SVM (Default), 1: Random, 2: NaiveBayes - unigrams, 3: NaiveBayes - bigrams).', type=int, default=0)
    args = parser.parse_args()
    start = time.time()
    # read training and test data
    if data.useGlove:
        data.readGloveData()

    data.read()


    evaluations = []

    for i in range(0,2):
    #for i in range(1):
        train = data.train[i]
        test = data.test[i]

        print('### Working on domain: ' + data.domains[i] + ' ###')

        # features that can be used by several feature extractors that we use for different classification tasks
        commonFeatures = features.Features(data.train[i])

        #print( ' here')

        #print(train);

        print('Train and predict aspects ...')
        ### classify aspects of a sentence ###
        aspectFeatureExtractor = aspectFeatures.AspectFeatures(train, commonFeatures)
        seqClassifier = svm.SVM(2)
        seqClassifier.train(train, aspectFeatureExtractor)
        seqClassifier.predict(test, aspectFeatureExtractor, 'preAspect')

        print('Train and predict aspect sentiments ...')
        ### classify aspect sentiments of a sentence ###
        aspectSentFeatureExtractor = aspectSentFeatures.AspectSentFeatures(train, commonFeatures)
        mcClassifier = svm.SVM(1)
        mcClassifier.train(train, aspectSentFeatureExtractor)
        mcClassifier.predict(test, aspectSentFeatureExtractor, 'preSentAsp')

        # Check if categories are defined for current set
        if len(train['categories']) > 0:
            print('Train and predict categories ...')
            for currCat in train['categories']:
                categoryFeatureExtractor = categoryFeatures.CategoryFeatures(train, commonFeatures, currCat)
                binClassifier = svm.SVM(0)
                binClassifier.train(train, categoryFeatureExtractor)
                binClassifier.predict(test, categoryFeatureExtractor, 'preCat')

            print('Train and predict category Sentiments ...')
            for currCat in train['categories']:
                catSentFeatureExtractor = categorySentFeatures.CategorySentFeatures(train, commonFeatures, currCat)
                mcClassifier = svm.SVM(1)
                mcClassifier.train(train, catSentFeatureExtractor)
                mcClassifier.predict(test, catSentFeatureExtractor, 'preCatSent')

        ### evaluate results ###
        evaluations.append(evaluation.evaluate(test))

    end = time.time()
    print("Elapsed time: " + str(round(end - start,1)) + "s")
    print()

    # number to percetage string
    def n2P (x): return str(round(100*x,2))

    # print final results
    print('####################################################################################')
    print('Summary of results:')
    # print aspect classification results
    print('### Aspect Classification ###')
    for i in range(len(evaluations)):
        asp = evaluations[i]['aspects']
        print('# ' + data.domains[i] + ': ' + n2P(asp['acc']) + '% \t(precision: ' +n2P(asp['prec']) + '%, recall: ' + n2P(asp['rec']) + '%, F1-score: ' + n2P(asp['f1']) + ')')

    # print aspect sentiment classification results
    print('### Aspect Sentiment Classification ###')
    for i in range(len(evaluations)):
        aspSent = evaluations[i]['aspSent']
        print('# ' + data.domains[i] + ':')
        for j in range(4):
            if aspSent[j]['acc'] != -1:
                print('# ' + data.sents[j] + ': ' + n2P(aspSent[j]['acc']) + '% \t(precision: ' + n2P(aspSent[j]['prec']) + '%, recall: ' + n2P(aspSent[j]['rec']) + '%, F1-score: ' + n2P(aspSent[j]['f1']) + ')')
            else:
                print('# ' + data.sents[j] + ': (Does not appear in test set)')
        print('# -> Average: ' + n2P(aspSent[4]['acc']) + '% \t(precision: ' + n2P(aspSent[4]['prec']) + '%, recall: ' + n2P(aspSent[4]['rec']) + '%, F1-score: ' + n2P(aspSent[4]['f1']) + ')')

    # print category classification results
    print('### Category Classification ###')
    for i in range(len(evaluations)):
        if 'category' in evaluations[i]:
            category = evaluations[i]['category']
            print('# ' + data.domains[i] + ':')
            j = 0
            for currCat in evaluations[i]['categories']:
                if category[j]['acc'] != -1:
                    print('# ' + currCat + ': ' + n2P(category[j]['acc']) + '% \t(precision: ' + n2P(category[j]['prec']) + '%, recall: ' + n2P(category[j]['rec']) + '%, F1-score: ' + n2P(category[j]['f1']) + ')')
                else:
                    print('# ' + currCat + ': (Does not appear in test set)')
                j += 1
            print('# -> Average: ' + n2P(category[j]['acc']) + '% \t(precision: ' + n2P(category[j]['prec']) + '%, recall: ' + n2P(category[j]['rec']) + '%, F1-score: ' + n2P(category[j]['f1']) + ')')

    # print category sentiment classification results
    print('### Category Sentiment Classification ###')
    for i in range(len(evaluations)):
        if 'category' in evaluations[i]:
            catSent = evaluations[i]['catSent']
            print('# ' + data.domains[i] + ':')
            j = 0
            for currCat in evaluations[i]['categories']:
                print('#    Category: ' + currCat + ':')
                # for each sentiment
                for k in range(4):
                    if catSent[j][k]['acc'] != -1:
                        print('#    ' + data.sents[k] + ': ' + n2P(catSent[j][k]['acc']) + '% \t(precision: ' + n2P(catSent[j][k]['prec']) + '%, recall: ' + n2P(catSent[j][k]['rec']) + '%, F1-score: ' + n2P(catSent[j][k]['f1']) + ')')
                    else:
                        print('#    ' + data.sents[k] + ': (Does not appear in test set)')
                print('#    -> Average: ' + n2P(catSent[j][4]['acc']) + '% \t(precision: ' + n2P(catSent[j][4]['prec']) + '%, recall: ' + n2P(catSent[j][4]['rec']) + '%, F1-score: ' + n2P(catSent[j][4]['f1']) + ')')
                j += 1

            print('# -> Average over all categories:')
            for k in range(4):
                print('# ' + data.sents[k] + ': ' + n2P(catSent[j][k]['acc']) + '% \t(precision: ' + n2P(catSent[j][k]['prec']) + '%, recall: ' + n2P(catSent[j][k]['rec']) + '%, F1-score: ' + n2P(catSent[j][k]['f1']) + ')')

    print('####################################################################################')

    sys.exit(0)