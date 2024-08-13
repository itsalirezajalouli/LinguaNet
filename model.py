#   Imports

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import unicodedata
import string
import torch.nn as nn
import requests, zipfile, io

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
        This method convers any input tring into a vector of long values,
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

