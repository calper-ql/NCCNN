import ctypes as ct
import struct
import numpy as np

'''  COMMANDS  '''

class SupplyItem:
    def __init__(self, image_id, classes):
        self.image_id = image_id
        self.classes = classes

    def generate_string(self):
        string = "+" + str(self.image_id)
        for c in self.classes:
            string += "_" + str(c)
        return string

def send_command(socket, cmd):
    socket.sendall(pack_command_size(cmd))
    socket.sendall(cmd)

def receive_command(socket):
    proto_data = socket.recv(4)
    if(proto_data):
        data = bytearray()
        size = unpack_command_size(proto_data)
        while size > 0:
            incoming = socket.recv(size)
            size -= len(incoming)
            data.extend(incoming)
        if data:
            return data
        else:
            raise error('ERROR: Client disconnected')
    else:
        raise error('ERROR: Client disconnected')

def pack_command_size(cmd):
    size = ct.c_uint(len(cmd))
    return bytearray(size)

def unpack_command_size(data):
    return struct.unpack('<I', data)[0]

def create_command(cmd):
    size = ct.c_uint(len(cmd))
    string = bytearray(size)
    string.extend(cmd.encode('ASCII'))
    return string

def unpack_command(cmd):
    size = struct.unpack('<I', cmd[:4])[0]
    string = cmd[4:4+size].decode()
    return string, cmd[4+size:]

########################

def pack_string(cmd, string):
    size = ct.c_uint(len(string))
    string_ = bytearray(size)
    string_.extend(string.encode('ASCII'))
    cmd.extend(string_)
    return cmd

def unpack_string(cmd):
    size = struct.unpack('<I', cmd[:4])[0]
    string = cmd[4:4+size].decode()
    return string, cmd[4+size:]

########################


def pack_unsigned(cmd, value):
    cnv = ct.c_uint(value)
    cmd.extend(cnv)
    return cmd


def unpack_unsigned(cmd):
    val = struct.unpack('<I', cmd[:4])[0]
    return val, cmd[4:]

########################


def pack_float(cmd, value):
    cnv = ct.c_float(value)
    cmd.extend(cnv)
    return cmd


def unpack_float(cmd):
    val = struct.unpack('<f', cmd[:4])[0]
    return val, cmd[4:]

########################


def pack_bool(cmd, value):
    cnv = ct.c_bool(value)
    cmd.extend(cnv)
    return cmd


def unpack_bool(cmd):
    val = struct.unpack('<?', cmd[:1])[0]
    return val, cmd[1:]

########################


def pack_np_array(cmd, value):
    array = value.tobytes()
    size = ct.c_uint(len(array))
    string = bytearray(size)
    string.extend(array)
    cmd.extend(string)
    return cmd


def unpack_np_array(cmd, dtype):
    size = struct.unpack('<I', cmd[:4])[0]
    buffer = np.frombuffer(cmd[4:4 + size], dtype=dtype, 
        count=size//np.dtype(dtype).itemsize)
    return buffer, cmd[4 + size:]

########################


''' END '''
