'This file contains a set of hyperparameters, that are used by several algorithms'

'Typical SVM HMM parameter C trading-off slack vs. magnitude of the weight-vector (without multiplying by the number of training samples)'
svmC_HMM = 0.76

'Typical SVM Multiclass parameter C trading-off slack vs. magnitude of the weight-vector'
svmC_MC = 0.1

'Number of most frequent words in a training set to be used as a unigram feature'
numUnigrams = 20000

'Number of most frequent bigrams in a training set to be used as a bigram feature'
numBigrams = 20000

'Size of the context window, thus how many words before resp. behin a word are considered as context words for context features'
contextWindow = 5

'Dimension of Word2Vec vectors, when computing centroid features'
w2vVecSize = 100