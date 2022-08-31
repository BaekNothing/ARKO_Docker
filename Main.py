import Consts
import UseData
import TrainData
import os

os.system("cls")
Consts.SetDisplay.ShowColoredText("\nwelcome to chatbot system ver 0.0.1", 'white')
Consts.SetDisplay.ShowColoredText("if you want to start Press any key\n", 'cyan')

input()
os.system("cls")
choosedNum = Consts.SetDisplay.SetSelectableScreen(["1. Train", "2. Use", "3. Exit"])

if choosedNum == 0 :
    TrainData.DoTrain()
elif choosedNum == 1 :
    UseData.StartModelToUse()

while input("this program will be closed... \nEnter \033[36m quit \033[37m to exit completely\n> ") == "" :
    os.system("cls")
    pass
exit()