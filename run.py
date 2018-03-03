#!/usr/bin/env python

# import the required packages here
import csv
import re

from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

import numpy as np
import tensorflow as tf


maxSeqLengthStatement = 25
maxSeqLengthSpeakerJob = 5
maxSeqLengthSpeaker = 5
maxSeqLengthSubject = 5
maxSeqLengthpartyAffiliation = 3
maxSeqLengthstateInfo = 1
maxSeqLengthcontext = 5

batchSize = 32
numClasses = 6
epochs = 1
keepProb = 0.9
numLayers = 2
initialLearningRate = 0.00001

inputLabels = {0:"pants-fire",1:"false",2:"barely-true",3:"half-true",4:"mostly-true",5:"true"}


def run(train_file, valid_file, test_file, output_file,embeddingFile="/tmp/glove/glove.6B.300d.txt"):
    try:
        vocab, embedding, dictionary, reverseDictionary = loadGlove(embeddingFile)
        vocabSize = len(vocab)
        embeddingSize = len(embedding[0])  ## 300
        embedding = np.asarray(embedding)
        vocab = np.asarray(vocab)
    except:
        print("embedding file not present")
        return

    parsedTrain = readTrainFile(train_file)
    parsedValid = readTrainFile(valid_file)
    parsedTest = readTestFile(test_file)

    ## Tokenising the statement file
    tokenisedStatementTrain = preProcessing(parsedTrain["statement"])
    ## getting the indices of the word.
    tokenisedStatementIndicesTrain = tokenIndices(tokenisedStatementTrain, dictionary, maxSeqLengthStatement)
    ## Output labels are converted into vectors
    outputLabelVectorsTrain = labelVectors(parsedTrain["label"])

    ## Valid
    tokenisedStatementValid = preProcessing(parsedValid["statement"])
    tokenisedStatementIndicesValid = tokenIndices(tokenisedStatementValid, dictionary, maxSeqLengthStatement)
    outputLabelVectorsValid = labelVectors(parsedValid["label"])

    ## Test
    tokenisedStatementTest = preProcessing(parsedTest["statement"])
    ## getting the indices of the word.
    tokenisedStatementIndicesTest = tokenIndices(tokenisedStatementTest, dictionary, maxSeqLengthStatement)

    ## Speaker Job
    tokenisedSpeakerJobTrain = preProcessing(parsedTrain["speakerJob"])
    tokenisedSpeakerJobIndicesTrain = tokenIndices(tokenisedSpeakerJobTrain, dictionary, maxSeqLengthSpeakerJob)

    tokenisedSpeakerJobValid = preProcessing(parsedValid["speakerJob"])
    tokenisedSpeakerJobIndicesValid = tokenIndices(tokenisedSpeakerJobValid, dictionary, maxSeqLengthSpeakerJob)

    tokenisedSpeakerJobTest = preProcessing(parsedTest["speakerJob"])
    tokenisedSpeakerJobIndicesTest = tokenIndices(tokenisedSpeakerJobTest, dictionary, maxSeqLengthSpeakerJob)

    ## Subject
    tokenisedSubjectTrain = preProcessing(parsedTrain["subject"])
    tokenisedSubjectIndicesTrain = tokenIndices(tokenisedSubjectTrain, dictionary, maxSeqLengthSubject)

    tokenisedSubjectValid = preProcessing(parsedValid["subject"])
    tokenisedSubjectIndicesValid = tokenIndices(tokenisedSubjectValid, dictionary, maxSeqLengthSubject)

    tokenisedSubjectTest = preProcessing(parsedTest["subject"])
    tokenisedSubjectIndicesTest = tokenIndices(tokenisedSubjectTest, dictionary, maxSeqLengthSubject)

    ## Party Affiliation
    tokenisedpartyAffiliationTrain = preProcessing(parsedTrain["partyAffiliation"])
    tokenisedpartyAffiliationIndicesTrain = tokenIndices(tokenisedpartyAffiliationTrain, dictionary,
                                                         maxSeqLengthpartyAffiliation)

    tokenisedpartyAffiliationValid = preProcessing(parsedValid["partyAffiliation"])
    tokenisedpartyAffiliationIndicesValid = tokenIndices(tokenisedpartyAffiliationValid, dictionary,
                                                         maxSeqLengthpartyAffiliation)

    tokenisedpartyAffiliationTest = preProcessing(parsedTest["partyAffiliation"])
    tokenisedpartyAffiliationIndicesTest = tokenIndices(tokenisedpartyAffiliationTest, dictionary,
                                                        maxSeqLengthpartyAffiliation)

    tf.reset_default_graph()
    trainModel(tokenisedStatementIndicesTrain=tokenisedStatementIndicesTrain,
               outputLabelVectorsTrain=outputLabelVectorsTrain,
               tokenisedStatementIndicesValid=tokenisedStatementIndicesValid,
               outputLabelVectorsValid=outputLabelVectorsValid,
               tokenisedSpeakerJobIndicesTrain=tokenisedSpeakerJobIndicesTrain,
               tokenisedSpeakerJobIndicesValid=tokenisedSpeakerJobIndicesValid,
               tokenisedpartyAffiliationIndicesTrain=tokenisedpartyAffiliationIndicesTrain,
               tokenisedpartyAffiliationIndicesValid=tokenisedpartyAffiliationIndicesValid,
               tokenisedSubjectIndicesTrain=tokenisedSubjectIndicesTrain,
               tokenisedSubjectIndicesValid=tokenisedSubjectIndicesValid,
               vocabSize=vocabSize, embeddingSize=embeddingSize, embedding=embedding, numLayers=numLayers,
               epochs=epochs)

    tf.reset_default_graph()
    outputPrediction = testModel(tokenisedStatementIndicesTest, tokenisedSpeakerJobIndicesTest,
                                 tokenisedSubjectIndicesTest, tokenisedpartyAffiliationIndicesTest, vocabSize,
                                 embeddingSize, embedding, numLayers)

    saveFile(outputPrediction, output_file)



## your implementation here

# read data from input


NEG_CONTRACTIONS = [
    (r'aren\'t', 'are not'),
    (r'can\'t', 'can not'),
    (r'couldn\'t', 'could not'),
    (r'daren\'t', 'dare not'),
    (r'didn\'t', 'did not'),
    (r'doesn\'t', 'does not'),
    (r'don\'t', 'do not'),
    (r'isn\'t', 'is not'),
    (r'hasn\'t', 'has not'),
    (r'haven\'t', 'have not'),
    (r'hadn\'t', 'had not'),
    (r'mayn\'t', 'may not'),
    (r'mightn\'t', 'might not'),
    (r'mustn\'t', 'must not'),
    (r'needn\'t', 'need not'),
    (r'oughtn\'t', 'ought not'),
    (r'shan\'t', 'shall not'),
    (r'shouldn\'t', 'should not'),
    (r'wasn\'t', 'was not'),
    (r'weren\'t', 'were not'),
    (r'won\'t', 'will not'),
    (r'wouldn\'t', 'would not'),
    (r'ain\'t', 'am not') # not only but stopword anyway
]
BLACKLIST_STOPWORDS = ['over','only','very','not','no']
ENGLISH_STOPWORDS = set(stopwords.words('english')) - set(BLACKLIST_STOPWORDS)
OTHER_CONTRACTIONS = {
    "'m": 'am',
    "'ll": 'will',
    "'s": 'has', # or 'is' but both are stopwords
    "'d": 'had'  # or 'would' but both are stopwords
}



def readTrainFile(file):
    with open(file,'r') as tsvin:
        tsvin = csv.reader(tsvin,delimiter ='\t')
        parsedFile = {"label" :[],"statement" :[],"subject" :[],"speaker":[],"speakerJob":[],"stateInfo":[],"partyAffiliation":[],"context":[]}
        for rowNum,row in enumerate(tsvin):
            try:
                ## Checking if all elements are accessible so all the elements are of same length
                a = [row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7]]
                parsedFile["label"].append(row[0])
                parsedFile["statement"].append(row[1])
                parsedFile["subject"].append(row[2])
                parsedFile["speaker"].append(row[3])
                parsedFile["speakerJob"].append(row[4])
                parsedFile["stateInfo"].append(row[5])
                parsedFile["partyAffiliation"].append(row[6])
                parsedFile["context"].append(row[7])
            except:
                print("Few inputs are in invalid format")
                #print(rowNum)
                #print(row)

        return parsedFile

def readTestFile(file):
    with open(file,'r') as tsvin:
        tsvin = csv.reader(tsvin,delimiter ='\t')
        parsedFile = {"statement" :[],"subject" :[],"speaker":[],"speakerJob":[],"stateInfo":[],"partyAffiliation":[],"context":[]}
        for rowNum,row in enumerate(tsvin):
            try:
                ## Checking if all elements are accessible so all the elements are of same length
                a = [row[0],row[1],row[2],row[3],row[4],row[5],row[6]]
                parsedFile["statement"].append(row[0])
                parsedFile["subject"].append(row[1])
                parsedFile["speaker"].append(row[2])
                parsedFile["speakerJob"].append(row[3])
                parsedFile["stateInfo"].append(row[4])
                parsedFile["partyAffiliation"].append(row[5])
                parsedFile["context"].append(row[6])
            except:
                print("Few inputs are in invalid format")
                print(rowNum)
                print(row)

        return parsedFile

# The input statement is expected a string.
def preProcessing(text,delimiter=' ',n=1):
    tokenisedOutput = []
    stemmer = PorterStemmer()
    for line in text:
        tokens = []

        ## Convert the line into lower case
        line = line.lower()

        ## Transform negative contractions
        for neg in NEG_CONTRACTIONS:
            line = re.sub(neg[0], neg[1], line)

        ## Tokenising the words
        tokens = word_tokenize(line)

        # transform other contractions (e.g 'll --> will)
        tokens = [OTHER_CONTRACTIONS[token] if OTHER_CONTRACTIONS.get(token)
                  else token for token in tokens]

        # removing punctuations, only retain words, no numbers and punctuation marks.
        r = r'[a-z]+'
        tokens = [word for word in tokens if re.search(r, word)]

#         # # remove irrelevant stop words
#         tokens = [token for token in tokens if token not in ENGLISH_STOPWORDS]

#         # stemming
#         tokens = [stemmer.stem(token) for token in tokens]


        ## Probably not required if using RNN for classification
        if n == 1:
            # return the list of words
            tokenisedOutput.append(tokens)
        else:
            # return the list of ngrams
            tokenisedOutput.append(ngrams(tokens, n))
        ##print(tokens)

    return tokenisedOutput

## Returns the indice of the statment which can be used for embedding lookup
def tokenIndices(text, dictionary, maxSeqLength=150):
    outputLength = len(text)
    tokenListIndices = np.zeros((outputLength, maxSeqLength))
    lineCount = 0
    tokenCount = 0

    for line in text:
        tokenCount = 0
        for token in line:
            try:
                tokenListIndices[lineCount][tokenCount] = dictionary[token]
            except:
                tokenListIndices[lineCount][tokenCount] = 399999
            tokenCount = tokenCount + 1
            if (tokenCount >= maxSeqLength):
                break
        lineCount = lineCount + 1

    return tokenListIndices

def labelVectors(labels):
    labelVectors = []
    defaultVectors = {"pants-fire":np.array([1,0,0,0,0,0]),"false":np.array([0,1,0,0,0,0]),"barely-true":np.array([0,0,1,0,0,0]),
                      "half-true":np.array([0,0,0,1,0,0]),"mostly-true":np.array([0,0,0,0,1,0]),"true":np.array([0,0,0,0,0,1])}
    for label in labels:
        labelVectors.append(defaultVectors[label])
    return np.asarray(labelVectors)

def loadGlove(embeddingFile):
    vocab = []
    embedding = []
    dictionary = {}
    reverseDictionary = {}
    count = 0
    file = open(embeddingFile, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embedding.append(row[1:])
        dictionary[row[0]] = count
        reverseDictionary[count] = row[0]
        count = count + 1
    print('Loaded GloVe!')
    file.close()
    return vocab, embedding,dictionary,reverseDictionary


# your training algorithm
def embeddingMatrix(sess,vocabSize,embeddingSize,embedding):
    W = tf.Variable(tf.constant(0.0, shape=[vocabSize, embeddingSize]),
                    trainable=False, name="W")
    embeddingPlaceholder = tf.placeholder(tf.float32, shape=[vocabSize, embeddingSize])
    embeddingInit = W.assign(embeddingPlaceholder)
    sess.run(embeddingInit, feed_dict={embeddingPlaceholder: embedding})
    return W

def getLstmCell(lstmUnits, keepProb):
    ## Basic LSTM cell is created
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    ## Dropout  wrapper is created
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=keepProb)
    return lstmCell

def build_model(input_data, embeddingMatrixWeights, numLayers, lstmUnits):
    ## embedding loookup for the input data from the embedding matrix
    data = tf.nn.embedding_lookup(embeddingMatrixWeights, input_data)

    ## Multi layer RNN
    lstmCell = tf.nn.rnn_cell.MultiRNNCell([getLstmCell(lstmUnits, keepProb) for _ in range(numLayers)],
                                           state_is_tuple=True)

    ## Dynamic rolling of LSTM
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
    ## Each cell will give us a output
    value = tf.transpose(value, [1, 0, 2])
    ## Only last layer is considered
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    ## Weights for the last layer to get a 6 dimensional vector
    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    ## Bias for each class
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    ## Prediction
    prediction = (tf.matmul(last, weight) + bias)
    return prediction

def getTrainBatch(index,batchSize,tokenisedStatementIndicesTrain,tokenisedSpeakerJobIndicesTrain,
                  tokenisedpartyAffiliationIndicesTrain,tokenisedSubjectIndicesTrain,outputLabelVectorsTrain):
    return tokenisedStatementIndicesTrain[index : index + batchSize] , tokenisedSpeakerJobIndicesTrain[index : index + batchSize],tokenisedpartyAffiliationIndicesTrain[index : index + batchSize] ,\
            tokenisedSubjectIndicesTrain[index : index + batchSize],np.array(outputLabelVectorsTrain[index:index+batchSize])


def trainModel(tokenisedStatementIndicesTrain, outputLabelVectorsTrain,
               tokenisedStatementIndicesValid, outputLabelVectorsValid,
               tokenisedSpeakerJobIndicesTrain, tokenisedSpeakerJobIndicesValid,
               tokenisedpartyAffiliationIndicesTrain, tokenisedpartyAffiliationIndicesValid,
               tokenisedSubjectIndicesTrain, tokenisedSubjectIndicesValid,
               vocabSize, embeddingSize, embedding, numLayers, epochs=10):
    trainingDataSize = len(tokenisedStatementIndicesTrain)
    prevMaxAccuracy = 0
    with tf.Session() as sess:
        embeddingMatrixWeights = embeddingMatrix(sess=sess, vocabSize=vocabSize, embeddingSize=embeddingSize,
                                                 embedding=embedding)
        ## Place holders for labels and input_data
        labels = tf.placeholder(tf.float32, [None, numClasses])
        print("Building Statement Model")
        with tf.variable_scope("statement"):
            ##Statement
            input_data_statement = tf.placeholder(tf.int32, [None, maxSeqLengthStatement])
            predictionStatement = build_model(input_data=input_data_statement,
                                              embeddingMatrixWeights=embeddingMatrixWeights,
                                              numLayers=numLayers, lstmUnits=maxSeqLengthStatement)
        print("Building Speaker Model")
        with tf.variable_scope("speakerJob"):
            ## speakerJob
            input_data_speakerJob = tf.placeholder(tf.int32, [None, maxSeqLengthSpeakerJob])
            predictionSpeakerJob = build_model(input_data=input_data_speakerJob,
                                               embeddingMatrixWeights=embeddingMatrixWeights,
                                               numLayers=numLayers, lstmUnits=maxSeqLengthSpeakerJob)
        print("party Affiliation Model")
        with tf.variable_scope("partyAffiliation"):
            ## party Affiliation
            input_data_partyAffiliation = tf.placeholder(tf.int32, [None, maxSeqLengthpartyAffiliation])
            predictionpartyAffiliation = build_model(input_data=input_data_partyAffiliation,
                                                     embeddingMatrixWeights=embeddingMatrixWeights,
                                                     numLayers=numLayers, lstmUnits=maxSeqLengthpartyAffiliation)

        with tf.variable_scope("Subject"):
            ## party Affiliation
            input_data_Subject = tf.placeholder(tf.int32, [None, maxSeqLengthSubject])
            predictionSubject = build_model(input_data=input_data_Subject,
                                            embeddingMatrixWeights=embeddingMatrixWeights,
                                            numLayers=numLayers, lstmUnits=maxSeqLengthSubject)

        predictionStatement = tf.reshape(predictionStatement, [-1, numClasses])
        wStatement = tf.Variable(tf.random_normal([numClasses, numClasses]))

        predictionSpeakerJob = tf.reshape(predictionSpeakerJob, [-1, numClasses])
        wSpeakerJob = tf.Variable(tf.random_normal([numClasses, numClasses]))

        predictionpartyAffiliation = tf.reshape(predictionpartyAffiliation, [-1, numClasses])
        wpartyAffiliation = tf.Variable(tf.random_normal([numClasses, numClasses]))

        predictionSubject = tf.reshape(predictionSubject, [-1, numClasses])
        wSubject = tf.Variable(tf.random_normal([numClasses, numClasses]))

        prediction = tf.matmul(predictionStatement, wStatement) + \
                     tf.matmul(predictionSpeakerJob, wSpeakerJob) + \
                     tf.matmul(predictionpartyAffiliation, wpartyAffiliation) + \
                     tf.matmul(predictionSubject, wSubject)
        ##prediction = predictionStatement

        correctPrediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        optimizer = tf.train.AdamOptimizer(initialLearningRate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for i in range(epochs):
            index = 0
            ## If data present is not exact multiple of batch size
            while index < trainingDataSize:
                if (index + batchSize <= trainingDataSize):
                    size = batchSize
                else:
                    size = trainingDataSize - index
                inputDataStatement, inputDataSpeakerJob, inputDataPartyAffiliation, inputDataSubject, outputData = getTrainBatch(
                    index, size, tokenisedStatementIndicesTrain,
                    tokenisedSpeakerJobIndicesTrain,
                    tokenisedpartyAffiliationIndicesTrain,
                    tokenisedSubjectIndicesTrain,
                    outputLabelVectorsTrain)
                sess.run(optimizer, feed_dict={input_data_statement: inputDataStatement,
                                               input_data_speakerJob: inputDataSpeakerJob,
                                               input_data_partyAffiliation: inputDataPartyAffiliation,
                                               input_data_Subject: inputDataSubject,
                                               labels: outputData})
                index = index + size

            currAccuracy = sess.run(accuracy, feed_dict={input_data_statement: tokenisedStatementIndicesValid,
                                                         input_data_speakerJob: tokenisedSpeakerJobIndicesValid,
                                                         input_data_partyAffiliation: tokenisedpartyAffiliationIndicesValid,
                                                         input_data_Subject: tokenisedSubjectIndicesValid,
                                                         labels: outputLabelVectorsValid})
            print("Current Accuracy : ", currAccuracy)
            if (currAccuracy > prevMaxAccuracy):
                prevMaxAccuracy = currAccuracy
                print("Written to model in epoch :", i + 1)
                save_path = saver.save(sess,"/tmp/model.ckpt")

        print("Model saved in path: %s" % save_path)

# your prediction code

def testModel(tokenisedStatementIndicesTest, tokenisedSpeakerJobIndicesTest, tokenisedSubjectIndicesTest,tokenisedpartyAffiliationIndicesTest, vocabSize,
              embeddingSize, embedding, numLayers):
    outputPrediction = []
    testDataSize = len(tokenisedStatementIndicesTest)
    with tf.Session() as sess:
        embeddingMatrixWeights = embeddingMatrix(sess=sess, vocabSize=vocabSize, embeddingSize=embeddingSize,
                                                 embedding=embedding)
        ## Place holders for labels and input_data
        labels = tf.placeholder(tf.float32, [None, numClasses])
        print("Building Statement Model")
        with tf.variable_scope("statement"):
            ##Statement
            input_data_statement = tf.placeholder(tf.int32, [None, maxSeqLengthStatement])
            predictionStatement = build_model(input_data=input_data_statement,
                                              embeddingMatrixWeights=embeddingMatrixWeights,
                                              numLayers=numLayers, lstmUnits=maxSeqLengthStatement)
        print("Building Speaker Model")
        with tf.variable_scope("speakerJob"):
            ## speakerJob
            input_data_speakerJob = tf.placeholder(tf.int32, [None, maxSeqLengthSpeakerJob])
            predictionSpeakerJob = build_model(input_data=input_data_speakerJob,
                                               embeddingMatrixWeights=embeddingMatrixWeights,
                                               numLayers=numLayers, lstmUnits=maxSeqLengthSpeakerJob)
        print("party Affiliation Model")
        with tf.variable_scope("partyAffiliation"):
            ## party Affiliation
            input_data_partyAffiliation = tf.placeholder(tf.int32, [None, maxSeqLengthpartyAffiliation])
            predictionpartyAffiliation = build_model(input_data=input_data_partyAffiliation,
                                                     embeddingMatrixWeights=embeddingMatrixWeights,
                                                     numLayers=numLayers, lstmUnits=maxSeqLengthpartyAffiliation)

        with tf.variable_scope("Subject"):
            ## party Affiliation
            input_data_Subject = tf.placeholder(tf.int32, [None, maxSeqLengthSubject])
            predictionSubject = build_model(input_data=input_data_Subject,
                                            embeddingMatrixWeights=embeddingMatrixWeights,
                                            numLayers=numLayers, lstmUnits=maxSeqLengthSubject)

        predictionStatement = tf.reshape(predictionStatement, [-1, numClasses])
        wStatement = tf.Variable(tf.random_normal([numClasses, numClasses]))
        # biasStatement = tf.Variable(tf.random_normal([numClasses]))

        predictionSpeakerJob = tf.reshape(predictionSpeakerJob, [-1, numClasses])
        wSpeakerJob = tf.Variable(tf.random_normal([numClasses, numClasses]))
        # biasSpeakerJob = tf.Variable(tf.random_normal([numClasses]))

        predictionpartyAffiliation = tf.reshape(predictionpartyAffiliation, [-1, numClasses])
        wpartyAffiliation = tf.Variable(tf.random_normal([numClasses, numClasses]))
        # biaspartyAffiliation = tf.Variable(tf.random_normal([numClasses]))

        predictionSubject = tf.reshape(predictionSubject, [-1, numClasses])
        wSubject = tf.Variable(tf.random_normal([numClasses, numClasses]))

        prediction = tf.matmul(predictionStatement, wStatement) + \
                     tf.matmul(predictionSpeakerJob, wSpeakerJob) + \
                     tf.matmul(predictionpartyAffiliation, wpartyAffiliation) + \
                     tf.matmul(predictionSubject, wSubject)

        # prediction = predictionStatement

        correctPrediction = tf.argmax(prediction, 1)

        saver = tf.train.Saver()
        saver.restore(sess,"/tmp/model.ckpt")
        outputPrediction.extend(
            sess.run(correctPrediction, feed_dict={input_data_statement: tokenisedStatementIndicesTest,
                                                   input_data_speakerJob: tokenisedSpeakerJobIndicesTest,
                                                   input_data_partyAffiliation: tokenisedpartyAffiliationIndicesTest,
                                                   input_data_Subject: tokenisedSubjectIndicesTest}))

    return outputPrediction


def saveFile(outputPrediction, fileName):
    f = open(fileName, 'w')
    for i in range(len(outputPrediction)):
        s = inputLabels[outputPrediction[i]]
        s = s + "\n"
        f.write(s)
    print("file written",fileName)
