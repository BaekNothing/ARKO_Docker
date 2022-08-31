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

koGPT2_TOKENIZER = GPT2TokenizerFast.from_pretrained("stable/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK, local_files_only=True) 
model = torch.load("stable/models/model.bin")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def SendStringToTorch(q, sent, a) : 
    return torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + sent + A_TKN + a)).unsqueeze(dim=0)

def ConvertIdsToTokens(pred) :
    return koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]

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
            if (int(inputStr) > 0 or int(inputStr) < len(lists)) :
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