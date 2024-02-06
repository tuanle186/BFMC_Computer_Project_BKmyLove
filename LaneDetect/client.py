import socket, threading
import time

class Client:
    def __init__(self, server_ip='192.168.0.101', server_port=12345):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        try:
            self.client_socket.connect((self.server_ip, self.server_port))
            print(f"Connected to server at {self.server_ip}:{self.server_port}")
        except: pass
    def client_send(self, data):
        # start_time = time.time()
        # with open('20_msg.txt','r') as file:
        #     for line in file:
        #         self.client_socket.sendall(line.encode())
        # end_time = time.time()
        # run_time = end_time - start_time
        # print("run time send 20 msg: ", run_time)
        # while True:
        #     msg = input("Client command: ")
        self.client_socket.sendall(data.encode())
    
    def client_recv(self):
        while True:
            try:
                data = self.client_socket.recv(1024).decode("utf-8").strip()
                print(data)
            except ConnectionResetError:
                break

if __name__ == "__main__":
    client = Client()
    client.connect()
    client_send_thread = threading.Thread(target=client.client_send)
    client_send_thread.start()
    client.client_recv()
