#   Imports

import torch
import string
import unicodedata
from idlmam import *
import torch.nn as nn
import requests, zipfile, io
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, random_split

#   Dataset pre-process

url = 'https://download.pytorch.org/tutorial/data.zip'
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

data = {}
allLetters = string.ascii_letters + " .,;'"                                         #   abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;
nLetters = len(allLetters)                                                          #   57
alphabet = {}

for i in range(nLetters):
    alphabet[allLetters[i]] = i                                                     #   maps every item to an integer(their index)

def unicode2Ascii(s):                                                               #   turns Unicode string to ASCII
    
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in allLetters
    )

for zipPath in z.namelist():                                                        #   loops through every language, opens the zip file & reades all the lines
    if 'data/names/' in zipPath and zipPath.endswith('.txt'):
        lang = zipPath[len('data/names/'):-len('.txt')]
        with z.open(zipPath) as zipFile:
            langNames = [unicode2Ascii(line).lower() for line in str(zipFile.read(), encoding = 'utf-8').strip().split('\n')]
            data[lang] = langNames
#         print(lang, ': ', len(langNames))

#   Dataset Object

class langNameDataset(Dataset):

    def __init__(self, langNameDict, vocabulary) -> None:
        self.labelNames = [x for x in langNameDict.keys()]
        self.data = []
        self.labels = []
        self.vocabulary = vocabulary
        for y, language in enumerate(self.labelNames):
            for sample in langNameDict[language]:
                self.data.append(sample)
                self.labels.append(y)

    def __len__(self):
        return len(self.data)

    def string2InputVec(self, inputString):
        '''
        This method converts any input string into a vector of long values,
        according to the vocabulary used by this object.
        inputString: the string to convert to a tensor
        '''
        T = len(inputString)
        nameVec = torch.zeros((T), dtype = torch.long)                              #   a long (64-bit integer) tensor to store the result

        for pos, character in enumerate(inputString):
            nameVec[pos] = self.vocabulary[character]

        return nameVec

    def __getitem__(self, idx):
        name = self.data[idx]
        label = self.labels[idx]
        labelVec = torch.tensor([label], dtype = torch.long)

        return self.string2InputVec(name), label

#   Train-test split

dataSet = langNameDataset(data, alphabet)

trainSplit, testSplit = random_split(dataSet, (len(dataSet) - 300, 300))

trainLoader = DataLoader(trainSplit, batch_size = 1, shuffle = True)
testLoader = DataLoader(testSplit, batch_size = 1, shuffle = False)

#   Last time step object

class lastTimeStep(nn.Module):
    '''
    A class for extracting the hidden activations of the last time step
    following the output of a PyTorch RNN module.
    '''

    def __init__(self, rnnLayers = 1, bidirectional = False) -> None:
        super(lastTimeStep, self).__init__()
        self.rnnLayers = rnnLayers
        if bidirectional:
            self.numDircetions = 2
        else:
            self.numDircetions = 1

    def forward(self, input):
        rnnOutput = input[0]                                                        #   result is either (out, ht) or (out, (ht, ct))

        lastStep = input[1]                                                         #   last step is ht
                                                                                    #   unless it's a tuple which then it's the first item in the tuple

        if (type(lastStep) == tuple):
            lastStep = input[0]
        
        batchSize = lastStep.shape[1]

        lastStep = lastStep.view(self.rnnLayers, self.numDircetions, batchSize, -1) #   per the docs, shape: (numLayers * numDirections, batch, hiddenSize )

        lastStep = lastStep[self.rnnLayers - 1]

        lastStep = lastStep.permute(1, 0, 2)

        return lastStep.reshape(batchSize, -1)                                      #   Flattens the last 2 dimentions into one (out, ht) or (out, (ht, ct))

#   Hyperparameters

dim = 64
vocabSize = len(allLetters)
hiddenNodes = 256
classes = len(dataSet.labelNames)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#   Model

rnnModel = nn.Sequential(
    nn.Embedding(vocabSize, dim),                                                   #   (vocab size, out dimention)
    nn.RNN(dim, hiddenNodes, batch_first = True),                                   #   (B, T, D) -> ((B, T, D), (S, B, D))
    lastTimeStep(),
    nn.Linear(hiddenNodes, classes),                                                #   classifier
)

lossFunc = nn.CrossEntropyLoss()
batchOneTrain = train_simple_network(rnnModel, lossFunc, trainLoader, testLoader, 
                                     score_funcs = {'Accuracy' : accuracy_score}, device = device, epochs = 5)

print(batchOneTrain)

string = '<Train finished!>'
print(f'{string:-^130}')

name = input('Choose a name: ')

predRnn = rnnModel.to('cpu').eval()
with torch.no_grad():
    preds = F.softmax(predRnn(
        dataSet.string2InputVec(name).reshape(1, -1)), dim = -1)
    for classId in range(len(dataSet.labelNames)):
        print(dataSet.labelNames[classId], ':',
              preds[0, classId].item() * 100)
