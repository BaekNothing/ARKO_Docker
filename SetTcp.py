import threading
import socket
from distutils.log import error
from tkinter import E
import Consts
import platform
import time
import os

def SetServer():
    server_socket = SetServerSocket()
    os.system("cls" if platform.system() == "Windows" else "clear")
    print("server Start : ")
    try:
        while True:
            try :
                client_socket, addr = server_socket.accept()
            except socket.timeout :
                continue

            serverThread = threading.Thread(target=binder, args=(client_socket, addr))
            serverThread.start()
            while serverThread.is_alive():
                serverThread.join(1)
    except KeyboardInterrupt:
        print("\033[31m mainThreadStop : by KeyboardInterrup \033[37m")
    except Exception as e:
        print("\033[31m mainThreadDown by :", e, '\033[37m')
    finally:
        server_socket.close()       

##############################################################################################
def SetServerSocket():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', 1280))
    server_socket.listen()
    server_socket.settimeout(0.5)
    return server_socket

def SendMessageToConnectedTarget(client_socket : socket, data :bytes) : 
    #Consts.SetDisplay.ShowColoredText("Message > [{}] sended".format(data.decode()), 'cyan')
    client_socket.sendall(len(data).to_bytes(4, byteorder='big'))
    client_socket.sendall(data)

def binder(client_socket, addr):
  try:
    startTime = time.time()
    msg = GetMessage(client_socket, addr)
    if len(msg) <= 1:
        client_socket.close()
        return

    data = "no mached key".encode()
    if msg[0] == 'C' :    
        data = SendMessageToChatBot(msg).encode()
    elif msg[0] == 'G' :
        data = SendMessageToTextGen(msg).encode()

    Consts.SetDisplay.ShowColoredText("sendTime : " + str(time.time() - startTime), 'yellow')
    SendMessageToConnectedTarget(client_socket, data)
  except Exception as e:
    print("\033[31m binderDown by : ", e, "from : ", addr, '\033[37m')
  finally:
    client_socket.close()



def GetMessage(client_socket, addr):
    data = client_socket.recv(4)
    length = int.from_bytes(data, "big")
    data = client_socket.recv(length)
    msg = data.decode('utf-8')
    print('Received from :', addr, msg)
    return msg

def SendMessageToChatBot(msg):
    sent = "0"  # 0=일상, 1=부정, 2=긍정
    answer = ""
    while 1:
        input_ids = Consts.SendStringToChatBot(msg, sent, answer)
        gen = Consts.ConvertIdsToTokens(Consts.chatbotModel(input_ids).logits)
        if gen == Consts.EOS:
            break
        answer += gen.replace("▁", " ")
    Consts.SetDisplay.ShowColoredText("Chatbot > {}".format(answer.strip()), 'cyan')
    return answer.strip()

def SendMessageToTextGen(msg):
    return Consts.SendStringToTextGen(msg)
