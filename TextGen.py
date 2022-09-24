import Consts

def SetTextGen() :
    
    # tokenizer.tokenize("안녕하세요. 한국어 GPT-2 입니다.😤:)l^o")
    # ['▁안녕', '하', '세', '요.', '▁한국어', '▁G', 'P', 'T', '-2', '▁입', '니다.', '😤', ':)', 'l^o']

    text = ""
    while True :
        while(text == "") :
            text = input("Enter text > ")
            if text == "exit" or text == "quit" :
                return 
        generated = Consts.SendStringToTextGen(text)
        print(generated)
        text = ""