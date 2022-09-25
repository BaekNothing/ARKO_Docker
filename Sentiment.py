import Consts

def SetSentiment() :
    text = ""
    while True:
        while(text == ""):
            text = input("Enter text > ")
            if text == "exit" or text == "quit":
                return
        generated = Consts.SendStringToSentiment(text)
        print(generated)
        text = ""