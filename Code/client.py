import socket
import numpy as np

soc = socket.socket()

soc.connect(('localhost', 999))

while True:
    inp = input("To Server:") 
    if inp != "close"   :
        mess = np.ones((1024, 1))
    else:
        mess = np.ones((1,1))
    
    # soc.send(bytes(mess, 'utf-8'))
    soc.send(mess.tobytes())
    
    mes_received = np.frombuffer(soc.recv(1024*8))
    print("From Server: ", len(mes_received))
    if len(mes_received) == 1:
        break
    # print("From Server: "+soc.recv(1024).decode())
    

soc.close()