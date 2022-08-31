import torch
import Consts
import os 

def StartModelToUse() :
    
    with torch.no_grad():
        print("input your question, if you want to exit, input 'exit' or 'quit'")
        while 1:
            q = ""
            while (q == "") :
                q = input("user > ").strip()
            if q == "quit" or q == "exit" :
                break

            sent = "0" # 0=일상, 1=부정, 2=긍정
            a = ""
            while 1:
                input_ids = Consts.SendStringToTorch(q, sent, a)
                gen = Consts.ConvertIdsToTokens(Consts.model(input_ids).logits)
                if gen == Consts.EOS:
                    break
                a += gen.replace("▁", " ")
            Consts.SetDisplay.ShowColoredText("Chatbot > {}".format(a.strip()), 'cyan')