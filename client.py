# telnet program example
import socket, select, string, sys
import threading
import time
from common import *
import cv2

class OIDClient(threading.Thread):
    # initialization
    def __init__(self, host ,port):
        host = host
        port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((host, port))
        threading.Thread.__init__(self)
        self.start()

    # when msg is being received
    def run(self):
        s = self.s
        recv_buffer = 4096
        print("Started client thread")

    def send_command(self, cmd):
        send_command(self.s, cmd)

    def request_supply_list(self):
        self.send_command(create_command("supply_list"))
        cmd = receive_command(self.s)
        string, cmd = unpack_command(cmd)
        print(string)

    def request_image_load(self, image_id):
        cmd = pack_string(create_command("load_image"), image_id)
        self.send_command(cmd)
        cmd = receive_command(self.s)
        string, cmd = unpack_command(cmd)
        return string

    def request_image_widthdraw(self, image_id):
        cmd = pack_string(create_command("widthdraw"), image_id)
        self.send_command(cmd)
        cmd = receive_command(self.s)
        string, cmd = unpack_command(cmd)
        if string == 'found':
            cmd = receive_command(self.s)
            string, cmd = unpack_command(cmd)
            w, cmd = unpack_unsigned(cmd)
            h, cmd = unpack_unsigned(cmd)
            image, cmd = unpack_np_array(cmd, np.uint8)
            image = np.reshape(image, [w, h, 3])
            return image
        else:
            return None



#main function
if __name__ == "__main__":

    s = OIDClient('192.168.1.31', 33333)

    print(s.request_image_load('f4d07a53ade71fea')); # 1
    print(s.request_image_load('e9af8dbca9e44c90')); # 1
    print(s.request_image_load('test_test_hahahaha')); # 0

    image = s.request_image_widthdraw('f4d07a53ade71fea')
    print(image.shape)

    cv2.imshow('f4d07a53ade71fea', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    time.sleep(0.03)
