import csv
import re

from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist

import numpy as np
import tensorflow as tf



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

maxSeqLength = 200
trainingFile = "/Users/sainikhilmaram/OneDrive/UCSB courses/Winter 2018/Deep Learning/HW2/liar_dataset/train.tsv"
embeddingFile = "/Users/sainikhilmaram/Desktop/glove/glove.6B.300d.txt"
testFile = "/Users/sainikhilmaram/OneDrive/UCSB courses/Winter 2018/Deep Learning/HW2/liar_dataset/test.tsv"

batchSize = 24
lstmUnits = 64
numClasses = 6
iterations = 10

inputLabels = {1:"pants-fire",2:"false",3:"barely-true",4:"half-true",5:"mostly-true",6:"true"}

def readTrainFile(file):
    with open(file,'r') as tsvin:
        tsvin = csv.reader(tsvin,delimiter ='\t')
        parsedFile = {"label" :[],"statement" :[],"subject" :[],"speaker":[],"speakerJob":[],"stateInfo":[],"partyAffiliation":[],"context":[]}
        for rowNum,row in enumerate(tsvin):
            try:
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

        # # remove irrelevant stop words
        # tokens = [token for token in tokens if token not in ENGLISH_STOPWORDS]

        # stemming
        #tokens = [stemmer.stem(token) for token in tokens]


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
def statementIndices(text, dictionary, outputLength):
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

def embeddingMatrix(sess,vocabSize,embeddingSize,embedding):
    W = tf.Variable(tf.constant(0.0, shape=[vocabSize, embeddingSize]),
                    trainable=False, name="W")
    embeddingPlaceholder = tf.placeholder(tf.float32, shape=[vocabSize, embeddingSize])
    embeddingInit = W.assign(embeddingPlaceholder)
    sess.run(embeddingInit, feed_dict={embeddingPlaceholder: embedding})
    return W

def getTrainBatch(index,batchSize,tokenisedStatementIndices,outputLabelVectors):
    return tokenisedStatementIndices[index : index + batchSize] , np.array(outputLabelVectors[index:index+batchSize])

def getTestBatch(index,batchSize,tokenisedStatementIndicesTest):
    return tokenisedStatementIndicesTest[index : index + batchSize]

def build_model(input_data,embeddingMatrixWeights):
    data = tf.nn.embedding_lookup(embeddingMatrixWeights,input_data)
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)
    return prediction

def trainModel(tokenisedStatementIndices, outputLabelVectors, vocabSize, embeddingSize, embedding, iterations=10):
    trainingDataSize = len(tokenisedStatementIndices)

    with tf.Session() as sess:
        embeddingMatrixWeights = embeddingMatrix(sess, vocabSize, embeddingSize, embedding)
        labels = tf.placeholder(tf.float32, [None, numClasses])
        input_data = tf.placeholder(tf.int32, [None, maxSeqLength])
        prediction = build_model(input_data, embeddingMatrixWeights)
        correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            index = 0
            ## If data present is not exact multiple of batch size
            while index < trainingDataSize:
                if (index + batchSize <= trainingDataSize):
                    size = batchSize
                else:
                    size = trainingDataSize - index
                inputData, outputData = getTrainBatch(index, size, tokenisedStatementIndices, outputLabelVectors)
                sess.run(optimizer, feed_dict={input_data: inputData, labels: outputData})
                index = index + size
        saver = tf.train.Saver()
        save_path = saver.save(sess,
                               "//Users/sainikhilmaram/OneDrive/UCSB courses/Winter 2018/Deep Learning/HW2/liar_dataset/model/model.ckpt")
        print("Model saved in path: %s" % save_path)

def testModel(tokenisedStatementIndicesTest,vocabSize,embeddingSize,embedding):
    outputPrediction = []
    testDataSize = len(tokenisedStatementIndicesTest)
    with tf.Session() as sess:
        embeddingMatrixWeights = embeddingMatrix(sess,vocabSize,embeddingSize,embedding)
        labels = tf.placeholder(tf.float32, [None, numClasses])
        input_data = tf.placeholder(tf.int32, [None, maxSeqLength])
        prediction = build_model(input_data,embeddingMatrixWeights)
        correctPrediction = tf.argmax(prediction,1)
        saver = tf.train.Saver()
        saver.restore(sess,"//Users/sainikhilmaram/OneDrive/UCSB courses/Winter 2018/Deep Learning/HW2/liar_dataset/model/model.ckpt")
        index = 0
        while index < testDataSize:
            if(index + batchSize <= testDataSize):
                size = batchSize
            else:
                size = testDataSize - index
            inputData = getTestBatch(index,batchSize,tokenisedStatementIndicesTest)
            outputPrediction.extend(sess.run(correctPrediction,feed_dict={input_data:inputData}))
            index = index + size
    return outputPrediction

def saveFile(outputPrediction, fileName):
    f = open(fileName, 'w')
    for i in range(len(outputPrediction)):
        s = inputLabels[outputPrediction[i]]
        s = s + "\n"
        f.write(s)


if __name__ == "__main__":
    print("Loading GLove")
    vocab, embedding, dictionary, reverseDictionary = loadGlove(embeddingFile)
    vocabSize = len(vocab)
    embeddingSize = len(embedding[0])  ## 300
    embedding = np.asarray(embedding)
    vocab = np.asarray(vocab)

    print("Reading Training File")
    parsedTraining = readTrainFile(trainingFile)
    ## Tokenising the statement file
    ##tokenisedStatement = preProcessing(["I shouldn't,have came here at 3","I'll be the Boss."])
    tokenisedStatement = preProcessing(parsedTraining["statement"])

    ## getting the indices of the word.
    tokenisedStatementIndices = statementIndices(tokenisedStatement, dictionary, len(tokenisedStatement))
    # print(tokenisedStatementIndices[0])

    ## Output labels are converted into vectors
    outputLabelVectors = labelVectors(parsedTraining["label"])

    print(len(outputLabelVectors))

    print("Reading Test File")
    parsedTest = readTestFile(testFile)
    tokenisedStatementTest = preProcessing(parsedTest["statement"])
    ## getting the indices of the word.
    tokenisedStatementIndicesTest = statementIndices(tokenisedStatementTest, dictionary, len(tokenisedStatementTest))

    print("Building the graph")
    tf.reset_default_graph()
    print("Training the model")
    trainModel(tokenisedStatementIndices, outputLabelVectors, vocabSize, embeddingSize, embedding, 1)

    print("Testing the model")
    tf.reset_default_graph()
    outputPrediction = testModel(tokenisedStatementIndicesTest, vocabSize, embeddingSize, embedding)

    print("Testing done")
    saveFile(outputPrediction, "mpredictions.txt")
    print("Save completed")



