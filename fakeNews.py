import csv
import re

from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist

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


# The input statement is expected a string.
def preProcessing(text,delimiter=' '):
    tokenisedOutput = []
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
        print(tokens)





if __name__ == "__main__":
    trainingFile = "/Users/sainikhilmaram/OneDrive/UCSB courses/Winter 2018/Deep Learning/HW2/liar_dataset/train.tsv"
    parsedTraining = readTrainFile(trainingFile)
    ## Tokenising the statement file
    ##preProcessing(parsedTraining["statement"])
    preProcessing(["I shouldn't have come here st 3","I'll be the Boss."])