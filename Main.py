import Consts
import UseData
import TrainData
import SetTcp
import platform
import os
import sys

if (sys.argv.__len__() >= 2 and sys.argv[1] == "RunServerDirectly") :
    SetTcp.SetServer()
else :
    os.system("cls" if platform.system() == "Windows" else "clear")
    Consts.SetDisplay.ShowColoredText("\nwelcome to chatbot system ver 0.0.1", 'white')
    Consts.SetDisplay.ShowColoredText("if you want to start Press any key\n", 'cyan')

    input()
    os.system("cls" if platform.system() == "Windows" else "clear")
    choosedNum = Consts.SetDisplay.SetSelectableScreen(["Train", "Talk", "Server", "Exit"])

    resultStr = ""
    if choosedNum == 0 :
        resultStr = TrainData.DoTrain()
    elif choosedNum == 1 :
        UseData.StartModelToUse()
    elif choosedNum == 2 :
        SetTcp.SetServer()
    while input(resultStr + "this program will be closed... \nEnter \033[36m quit \033[37m to exit completely\n> ") == "" :
        os.system("cls" if platform.system() == "Windows" else "clear")
        pass
exit()