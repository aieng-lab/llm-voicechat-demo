import socket
import numpy as np

ser = socket.socket()

ser.bind(('localhost', 999))

ser.listen(3)
closed = False
mes_received = ""
while not closed:
    
    c, adress = ser.accept()
    # mes_received = c.recv(1024).decode()
    mes_received = np.frombuffer(c.recv(1024*8))
    print('From client : ', len(mes_received))
    while True:
        if len(mes_received) != 1:
            # mes_sent = input('To Client:')
            # c.send(bytes(mes_sent, 'utf-8'))
            mes_sent = np.ones((1024, 1))
            c.send(mes_sent.tobytes())
            # mes_received = c.recv(1024).decode()
            mes_received = np.frombuffer(c.recv(1024*8))
            print('From client : ' , len(mes_received))
        else:
            closed = True
            mes_sent = np.ones((1, 1))
            c.send(mes_sent.tobytes())
            break
        

c.close()