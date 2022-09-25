import os
import time
import math
import platform
from regex import D
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers import pipeline
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from transformers import GPT2TokenizerFast


Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

#load ChatBotModel
chatbotPath = 'stable/models/'
chatbotName = 'model.bin'
koGPT2_TOKENIZER = GPT2TokenizerFast.from_pretrained("stable/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK, local_files_only=True) 
chatbotModel = torch.load("stable/models/model.bin")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chatbotModel.to(device)

def SwitchChatBotModel(path) : 
    global chatbotModel
    chatbotModel = torch.load(path)
    chatbotModel.to(device)
    return chatbotModel

def SendStringToChatBot(q, sent, a) : 
    return torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + sent + A_TKN + a)).unsqueeze(dim=0)

def ConvertIdsToTokens(inputModel) :
    return koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(inputModel, dim=-1).squeeze().numpy().tolist())[-1]

#load TextGen 
textGenPath = "./stable/textGenModels/"
textGenName = "pytorch_model.bin"
if not os.path.exists(textGenPath + textGenName):
    textGenModel = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    textGenModel.save_pretrained(textGenPath)
else:
    textGenModel = GPT2LMHeadModel.from_pretrained(textGenPath)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
textGenModel.to(device)

#load Sentiment
sentPath = "./stable/sentiment/"
if not os.path.exists(sentPath):
    sentClassifier_ko = pipeline(
        task="sentiment-analysis", device=-1, model="nlptown/bert-base-multilingual-uncased-sentiment")
    os.mkdir(sentPath)
    sentClassifier_ko.save_pretrained(sentPath)
else:
    sentClassifier_ko = pipeline(
        task="sentiment-analysis", device=-1, model=sentPath)

def SendStringToSentiment(msg : str) -> str :
    return str(sentClassifier_ko(msg)[0])

def SendStringToTextGen(msg : str) -> str :
    input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(msg)).unsqueeze(dim=0)
    output = textGenModel.generate(input_ids,
                                max_length=128,
                                repetition_penalty=2.0,
                                pad_token_id=koGPT2_TOKENIZER.pad_token_id,
                                eos_token_id=koGPT2_TOKENIZER.eos_token_id,
                                bos_token_id=koGPT2_TOKENIZER.bos_token_id,
                                use_cache=True)
    return koGPT2_TOKENIZER.decode(output[0])


class Math() :
    def Clamp(x, min, max) :
        return max if x > max else min if x < min else x

class SetDisplay() :
    def SetSelectableScreen(lists : list, point  = 0) :
        inputStr = ""
        while(not lists.__contains__(inputStr)) :
            os.system("cls" if platform.system() == "Windows" else "clear")
            for i in range(len(lists)):
                print(i, ".", lists[i])
            SetDisplay.ShowColoredText(
                "Enter items correctly or numbers for selection", 'cyan')
            inputStr = input("\n" + "> ")
            if (int(inputStr) >= 0 or int(inputStr) < len(lists)) :
                return int(inputStr)
        return lists.index(inputStr)

    def ShowColoredText(text : str, color : str):
        SetDisplay.ChangeColor(color)
        print(text)
        SetDisplay.ChangeColor('white')

    def ChangeColor(code : str):
        
        colors = {
            'black': 30,
            'red': 31,
            'green': 32,
            'yellow': 33,
            'blue': 34,
            'magenta': 35,
            'cyan': 36,
            'white': 37,
        }
        print('\033['+ str(colors[code]) +'m', end = '\r')