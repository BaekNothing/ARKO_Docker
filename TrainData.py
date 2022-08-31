import platform
import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import DataLoader, Dataset
import sys
import os
import os.path
import datetime as dt
import Consts
import multiprocessing as mp
from multiprocessing import freeze_support


# 챗봇 데이터를 처리하는 클래스를 만든다.
class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=40):  # 데이터셋의 전처리를 해주는 부분
        self._data = chats
        self.max_len = max_len
        self.q_token = Consts.Q_TKN
        self.a_token = Consts.A_TKN
        self.sent_token = Consts.SENT
        self.eos = Consts.EOS
        self.mask = Consts.MASK
        self.tokenizer = Consts.koGPT2_TOKENIZER

    def __len__(self):  # chatbotdata 의 길이를 리턴한다.
        return len(self._data)

    def __getitem__(self, idx):  # 로드한 챗봇 데이터를 차례차례 DataLoader로 넘겨주는 메서드
        turn = self._data.iloc[idx]
        q = turn["Q"]  # 질문을 가져온다.
        q = re.sub(r"([?.!,])", r" ", q)  # 구둣점들을 제거한다.

        a = turn["A"]  # 답변을 가져온다.
        a = re.sub(r"([?.!,])", r" ", a)  # 구둣점들을 제거한다.

        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        q_len = len(q_toked)

        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)

        #질문의 길이가 최대길이보다 크면
        if q_len > self.max_len:
            a_len = self.max_len - q_len        #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:       #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_toked = q_toked[-(int(self.max_len / 2)) :]   #질문길이를 최대길이의 반으로 
                q_len = len(q_toked)
                a_len = self.max_len - q_len              #답변의 길이를 최대길이 - 질문길이
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        #질문의 길이 + 답변의 길이가 최대길이보다 크면
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len        #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:       #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_toked = q_toked[-(int(self.max_len / 2)) :]   #질문길이를 최대길이의 반으로 
                q_len = len(q_toked)
                a_len = self.max_len - q_len              #답변의 길이를 최대길이 - 질문길이
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        # 답변 labels = [mask, mask, ...., mask, ..., <bos>,..답변.. <eos>, <pad>....]
        labels = [self.mask,] * q_len + a_toked[1:]

        # mask = 질문길이 0 + 답변길이 1 + 나머지 0
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        # 답변 labels을 index 로 만든다.
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        # 최대길이만큼 PADDING
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        # 질문 + 답변을 index 로 만든다.    
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        # 최대길이만큼 PADDING
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        #질문+답변, 마스크, 답변
        return (token_ids, np.array(mask), labels_ids)

def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(np.array(data)), torch.LongTensor(np.array(mask)), torch.LongTensor(np.array(label))

def DoTrain() :
    modelPath = SetModelPath()
    model = torch.load(modelPath)
    Chatbot_Data = SetChatBotData()
    overWriteFlag = Consts.SetDisplay.SetSelectableScreen(["OverWrite data", "Save new data"])
    
    #if window num_workers is 0 and linux num_workers is 2
    
    device = SetDevice()
    model.to(device)
    model.train()

    epoch = 10
    print ("train start")
    for epoch in range(epoch):
        print ("epoch : ", epoch)
        printProgressBar(epoch, 10, prefix = 'Progress:', suffix = 'Complete', decimals=10, barLength = 50)
        model = TrainEachData(model, Chatbot_Data)

    return SaveModel(model, modelPath, overWriteFlag)

def SetModelPath() :
    modelList = list(filter(lambda x: x.__contains__('model'), os.listdir("./stable/models/")))
    modelIndex = Consts.SetDisplay.SetSelectableScreen(modelList)
    modelPath = "./stable/models/" + modelList[modelIndex]
    return modelPath

def SetChatBotData() :
    Chatbot_DataList = os.listdir("./stable/data/")
    Chatbot_DataIndex = Consts.SetDisplay.SetSelectableScreen(Chatbot_DataList)
    Chatbot_Data = pd.read_csv(
        "./stable/data/" + Chatbot_DataList[Chatbot_DataIndex])
    Chatbot_Data = Chatbot_Data[:30]
    Chatbot_Data.head()
    return Chatbot_Data

def SetDevice() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def SetTrainDataLoader(Chatbot_Data: pd.DataFrame):
    workers = 0 if platform.system() == "Windows" else 2
    train_set = ChatbotDataset(Chatbot_Data, max_len=40)
    train_dataloader = DataLoader(
        train_set, batch_size=32, num_workers=workers, shuffle=True, collate_fn=collate_batch)
    return train_dataloader

def TrainEachData(model, Chatbot_Data: pd.DataFrame) :
    train_dataloader = SetTrainDataLoader(Chatbot_Data)
    learning_rate = 3e-5
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(Consts.model.parameters(), lr=learning_rate)
    Sneg = -1e18
    for batch_idx, samples in enumerate(train_dataloader):
        optimizer.zero_grad()
        token_ids, mask, label = samples
        out = model(token_ids)
        out = out.logits
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(
            repeats=out.shape[2], dim=2)
        mask_out = torch.where(
            mask_3d == 1, out, Sneg * torch.ones_like(out))
        loss = criterion(mask_out.transpose(2, 1), label)
        avg_loss = loss.sum() / mask.sum()
        avg_loss.backward()
        optimizer.step()
    return model

def SaveModel(model, modelPath, overWriteFlag) :
    resultString = "this model is saved as "
    if(overWriteFlag == 0):
        torch.save(model, modelPath)
        resultString += modelPath + '\n'
    else:
        now = dt.datetime.now()
        nowDate = now.strftime('%m%d_%H%M')
        # torch.save(model.state_dict(), './stable/models/' +
        #            str(nowDate) + '_state_dict.bin')
        torch.save(model, './stable/models/' + str(nowDate) + '_model.bin')
        resultString += str(nowDate) + '_model.bin\n'
    resultString += "train end\n"
    return resultString

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    filledLength = int(round(barLength * iteration / float(total)))
    percents = round(100.00 * (iteration / float(total)), decimals)
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        print("\n")