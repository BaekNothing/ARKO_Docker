# from https://wikidocs.net/157001 KoGPT2 챗봇 만들기

import math
import numpy as np
import pandas as pd
import random
import re
import torch
import urllib.request
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast
from transformers import GPT2TokenizerFast
import os

BOS = "</s>"
EOS = "</s>"
PAD = "<pad>"
MASK = "<unused0>"

print(os.getcwd())

# 허깅페이스 transformers 에 등록된 사전 학습된 koGTP2 토크나이저를 가져온다.
koGPT2_TOKENIZER = GPT2TokenizerFast.from_pretrained("../stable/kogpt2-base-v2",
                                                           bos_token=BOS, eos_token=EOS, unk_token='<unk>',
                                                     pad_token=PAD, mask_token=MASK, local_files_only=True)
# bos_token : 문장의 시작을 나타내는 token
# eos_token : 문장의 끝을 나타내는 token
# unk_token : 모르는 단어를 나타내는 token
# pad_token : 동일한 batch 내에서 입력의 크기를 동일하게 하기 위해서 사용해는 token

# PreTrainedTokenizer 에서 제공되는 함수는
# tokenize() : tokenizer를 이용해서 string을 token id의 리스트로 변환한다.
# get_added_vocab() : token to index에 해당하는 dict를 리턴한다.
# batch_decode() : token id로 구성된 입력을 하나의 연결된 string으로 출력한다.
# convert_ids_to_tokens() : token id 의 리스트를 token으로 변환한다. skip_special_tokens=True로 하면 decoding할 때 special token들을 제거한다.
# convert_tokens_to_ids() : token string의 리스트를 token id 또는 Token id의 리스트로 변환한다.
# decode() : tokenizer 와 vocabulary를 이용해서 token id를 string으로 변환한다. skip_special_token=True로 지정하면 speical token들을 제외한다.
# encode() : token string을 token id 의 리스트로 변환한다. add_special_tokens=False로 지정하면 token id로 변환할 때 special token들을 제외한다. padding을 통해서 padding token을 어떻게 추가할지조 지정할 수 있다.

Q_TKN = "<usr>"
A_TKN = "<sys>"
SENT = '<unused1>'

# 챗봇 데이터를 처리하는 클래스를 만든다.
class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=40):  # 데이터셋의 전처리를 해주는 부분
        self._data = chats
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.sent_token = SENT
        self.eos = EOS
        self.mask = MASK
        self.tokenizer = koGPT2_TOKENIZER

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

# 리턴되는 데이터는 token_ids, mask, labels_ids 입니다. token_ids 는 + 질문문장 + + 감정 + + 답변 + + pad_token_id 순서 입니다. pad_token_id는 max_len 에 일치하도록 추가 됩니다. mask 는 질문 q가 들어 가는 곳에는 0, 답변 a가 위치한 곳에는 1 그리고 빈 공간에는 0 으로 채워 집니다. labels은 질문의 길이만큼 mask 문자 그리고 답변 a의 id 입니다.
# 배치 데이터를 만들기 위해 collate_batch 함수를 정의합니다.

def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)
