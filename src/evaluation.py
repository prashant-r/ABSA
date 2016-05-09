
def measurements(truePos,trueNeg,falsePos,falseNeg):

    accuracy = 0;
    recall = 0;
    precision = 0;
    f1 = 0;

    if((truePos + trueNeg + falsePos + falseNeg)!=0):
        accuracy = (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)

    if(truePos+ falsePos != 0):
        precision = truePos / (truePos + falsePos)

    if(truePos+ falseNeg != 0):
        recall = truePos / (truePos + falseNeg)

    if((precision + recall)!=0):
        f1 = 2 * precision*recall / (precision + recall)

    return [accuracy, recall, precision, f1]

'compute accuracy and other measurements of a test set'
def evaluate(test):

    evaluation = {}

    # aspect
    aspTruePos = 0
    aspFalsePos = 0
    aspTrueNeg = 0
    aspFalseNeg= 0

    # aspect sentiment
    aspSentTruePos = [0]*4
    aspSentFalsePos = [0]*4
    aspSentTrueNeg = [0]*4
    aspSentFalseNeg= [0]*4

    # category
    numCat = len(test['categories']) # number of different categories

    catTruePos = [0] * numCat
    catFalsePos = [0] * numCat
    catTrueNeg = [0] * numCat
    catFalseNeg= [0] * numCat

    # category sentiments
    catSentTruePos = [[0]*4 for i in range(numCat)]
    catSentFalsePos = [[0]*4 for i in range(numCat)]
    catSentTrueNeg = [[0]*4 for i in range(numCat)]
    catSentFalseNeg = [[0]*4 for i in range(numCat)]

    # add some sparsity checks: thus, if the label does not appear in the test set, then there is no sense in computing measurements for it
    aspSentSparsity = [False] * 4
    catSparsity = [False] * numCat
    catSentSparsity = [[False]*4 for i in range(numCat)]
    
    # for each discussion
    for sentence in test['sentences']:

        # for each aspect
        for i in range(len(sentence['aspects'])):
            asp = sentence['aspects'][i]
            pasp = sentence['preAspect'][i]

            if asp > 0 and pasp > 0:
                aspTruePos += 1
                
            elif asp > 0 and pasp == 0:
                aspFalseNeg += 1
                
            elif asp == 0 and pasp > 0:
                aspFalsePos += 1
                
            else: # asp == 0 and pasp == 0
                aspTrueNeg += 1


            aspSent = sentence['aspects'][i]
            paspSent = sentence['preSentAsp'][i]

            for i in range(4):
                if aspSent == i+1:
                    aspSentSparsity[i] = True

                if aspSent == i+1 and paspSent == i+1:
                    aspSentTruePos[i] += 1

                elif aspSent == i+1 and paspSent != i+1:
                    aspSentFalseNeg[i] += 1

                elif aspSent != i+1 and paspSent == i+1:
                    aspSentFalsePos[i] += 1

                else: # aspSent != i+1 and paspSent != i+1
                    aspSentTrueNeg[i] += 1

        # for each category
        i = 0
        for currCat in test['categories']:
            isCat = True if 'polarity' in sentence['categories'][currCat] else False # sentence has category
            isPred = True if sentence['categories'][currCat]['preCat'] == 1 else False # category was predicted for sentence

            if isCat:
                catSparsity[i] = True

            if isCat and isPred:
                catTruePos[i] += 1

            elif isCat and not isPred:
                catFalseNeg[i] += 1

            elif not isCat and isPred:
                catFalsePos[i] += 1

            else: # not isCat and not isPred
                catTrueNeg[i] += 1

            if isCat:
                # for each sentiment
                for j in range(4):
                    isPol = 1 if sentence['categories'][currCat]['polarity'] == j+1 else 0
                    predPol = 1 if sentence['categories'][currCat]['preCatSent'] == j+1 else 0

                    if isPol:
                        catSentSparsity[i][j] = True

                    if isPol and predPol:
                        catSentTruePos[i][j] += 1

                    elif isPol and not predPol:
                        catSentFalseNeg[i][j] += 1

                    elif not isPol and predPol:
                        catSentFalsePos[i][j] += 1

                    else: # not isPol and not predPol
                        catSentTrueNeg[i][j] += 1

            i += 1

    aspects = {}
    [aspects['acc'], aspects['rec'], aspects['prec'], aspects['f1']] = measurements(aspTruePos, aspTrueNeg, aspFalsePos, aspFalseNeg)
    evaluation['aspects'] = aspects

    aspSent = [None]*5
    aspSent[4] = {} # average measurements
    aspSent[4]['acc'] = 0
    aspSent[4]['prec'] = 0
    aspSent[4]['rec'] = 0
    aspSent[4]['f1'] = 0

    avgNum = 0

    for i in range(4):
        aspSent[i] = {}

        if aspSentSparsity[i] == False:
            aspSent[i]['acc'] = -1

        else:
            [aspSent[i]['acc'], aspSent[i]['rec'], aspSent[i]['prec'], aspSent[i]['f1']] = measurements(aspSentTruePos[i], aspSentTrueNeg[i], aspSentFalsePos[i], aspSentFalseNeg[i])

            aspSent[4]['acc'] += aspSent[i]['acc']
            aspSent[4]['prec'] += aspSent[i]['prec']
            aspSent[4]['rec'] += aspSent[i]['rec']
            aspSent[4]['f1'] += aspSent[i]['f1']

            avgNum += 1

    aspSent[4]['acc'] /= avgNum
    aspSent[4]['prec'] /= avgNum
    aspSent[4]['rec'] /= avgNum
    aspSent[4]['f1'] /= avgNum


    evaluation['aspSent'] = aspSent

    if numCat > 0:
        evaluation['categories'] = test['categories']

        catAvgNum = 0

        # category
        category = [None]*(numCat+1) # plus one for averafe
        category[numCat] = {} # average measurements
        category[numCat]['acc'] = 0
        category[numCat]['prec'] = 0
        category[numCat]['rec'] = 0
        category[numCat]['f1'] = 0
        for i in range(numCat):
            category[i] = {}

            if catSparsity[i] == False:
                category[i]['acc'] = -1

            else:
                [category[i]['acc'], category[i]['rec'], category[i]['prec'], category[i]['f1']] = measurements(catTruePos[i], catTrueNeg[i], catFalsePos[i], catFalseNeg[i])

                category[numCat]['acc'] += category[i]['acc']
                category[numCat]['prec'] += category[i]['prec']
                category[numCat]['rec'] += category[i]['rec']
                category[numCat]['f1'] += category[i]['f1']

                catAvgNum += 1

        category[numCat]['acc'] /= catAvgNum
        category[numCat]['prec'] /= catAvgNum
        category[numCat]['rec'] /= catAvgNum
        category[numCat]['f1'] /= catAvgNum

        evaluation['category'] = category

        # category sentiment
        catSent = [[None]*5 for i in range(numCat+1)] # plus one for average

        catSentComplAvgNum = [0]*4

        for j in range(4): # init overall average
            catSent[numCat][j] = {} # average measurements
            catSent[numCat][j]['acc'] = 0
            catSent[numCat][j]['prec'] = 0
            catSent[numCat][j]['rec'] = 0
            catSent[numCat][j]['f1'] = 0

        i = 0
        for currCat in test['categories']:
            catSent[i][4] = {} # average measurements
            catSent[i][4]['acc'] = 0
            catSent[i][4]['prec'] = 0
            catSent[i][4]['rec'] = 0
            catSent[i][4]['f1'] = 0

            catSentAvgNum = 0

            for j in range(4):
                catSent[i][j] = {}

                if catSentSparsity[i][j] == False:
                    catSent[i][j]['acc'] = -1

                else:
                    [catSent[i][j]['acc'], catSent[i][j]['rec'], catSent[i][j]['prec'], catSent[i][j]['f1']] = measurements(catSentTruePos[i][j], catSentTrueNeg[i][j], catSentFalsePos[i][j], catSentFalseNeg[i][j])

                    catSent[i][4]['acc'] += catSent[i][j]['acc']
                    catSent[i][4]['prec'] += catSent[i][j]['prec']
                    catSent[i][4]['rec'] += catSent[i][j]['rec']
                    catSent[i][4]['f1'] += catSent[i][j]['f1']

                    catSent[numCat][j]['acc'] += catSent[i][j]['acc']
                    catSent[numCat][j]['prec'] += catSent[i][j]['prec']
                    catSent[numCat][j]['rec'] += catSent[i][j]['rec']
                    catSent[numCat][j]['f1'] += catSent[i][j]['f1']

                    catSentAvgNum += 1
                    catSentComplAvgNum[j] += 1

            catSent[i][4]['acc'] /= catSentAvgNum
            catSent[i][4]['prec'] /= catSentAvgNum
            catSent[i][4]['rec'] /= catSentAvgNum
            catSent[i][4]['f1'] /= catSentAvgNum

            i += 1

        for j in range(4):
            catSent[numCat][j]['acc'] /= catSentComplAvgNum[j]
            catSent[numCat][j]['prec'] /= catSentComplAvgNum[j]
            catSent[numCat][j]['rec'] /= catSentComplAvgNum[j]
            catSent[numCat][j]['f1'] /= catSentComplAvgNum[j]

        evaluation['catSent'] = catSent
                
    return evaluation