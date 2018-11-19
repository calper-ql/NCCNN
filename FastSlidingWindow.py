import numpy as np
import cv2
from client import *

def sliding_axis_test(axis_size, stride, size):
    value = ((axis_size // stride) * stride + size) - axis_size 
    return value

def pad_image_to_fit_windows(image, stride, size):
    width_test = sliding_axis_test(image.shape[0], stride, size)
    height_test = sliding_axis_test(image.shape[1], stride, size)
    zeros = np.zeros([image.shape[0] + width_test, image.shape[1] + height_test, image.shape[2]])
    zeros[width_test//2:width_test//2 + image.shape[0], height_test//2:height_test//2+image.shape[1]] = image
    return zeros / 255.0

def generate_padded_index_map(image_shape, stride, size):
    width_test = sliding_axis_test(image_shape[0], stride, size)
    height_test = sliding_axis_test(image_shape[1], stride, size)
    width_size = ((image_shape[0]+width_test) // stride) - size//stride + 1
    height_size = ((image_shape[1]+height_test) // stride) - size//stride + 1
    idx_map = np.zeros([width_size, height_size, 2], dtype=np.int32)
    width_arrangement = np.arange(width_size) * (stride)
    height_arrangement = np.arange(height_size) * (stride)
    idx_map[:, :, 0] = width_arrangement.reshape([width_arrangement.shape[0], 1])
    idx_map[:, :, 1] = height_arrangement.reshape([1, height_arrangement.shape[0]]) 
    return idx_map

'''
Generates a map with shape of [pad_height, pad_width, 3]
the values are the coordinates within 0 - 1 to the center of the patches 
+ the size relative to the image shape

'''
def generate_coordinate_map(image_shape, stride, size):
    width_test = sliding_axis_test(image_shape[0], stride, size)
    height_test = sliding_axis_test(image_shape[1], stride, size)
    width = image_shape[0] + width_test
    height = image_shape[1] + height_test
    idx_map = generate_padded_index_map(image_shape, stride, size)
    cmap = np.array(idx_map.copy(), dtype=np.float32)
    cmap[:, :, 0] = idx_map[:, :, 0] / image_shape[0]
    cmap[:, :, 1] = idx_map[:, :, 1] / image_shape[1]
    cmap[:, :, 0] -= (width_test/2)/image_shape[0]
    cmap[:, :, 1] -= (height_test/2)/image_shape[1]
    cmap[:, :, 0] += (size/2)/image_shape[0]
    cmap[:, :, 1] += (size/2)/image_shape[1]
    return cmap


def sliding_window(image, stride, size):
    padded = pad_image_to_fit_windows(image, stride, size)
    idx_map = generate_padded_index_map(image.shape, stride, size)
    def range_extract(x):
        return padded[x[0]:x[0]+size, x[1]:x[1]+size]
    idx_map_flat = np.reshape(idx_map, [idx_map.shape[0] * idx_map.shape[1], 2])
    extracted = np.apply_along_axis(range_extract, axis=1, arr=idx_map_flat)
    extracted = np.reshape(extracted, [idx_map.shape[0], idx_map.shape[1], extracted.shape[1], extracted.shape[2], extracted.shape[3]])
    cmap = generate_coordinate_map(image.shape, stride, size)
    return extracted, cmap

def shifted_reconstruct(extracted):
    r = np.zeros([extracted.shape[0]*extracted.shape[2], extracted.shape[1]*extracted.shape[3], extracted.shape[4]])
    def set_extract(x): 
        r[  x[0]*extracted.shape[2]:x[0]*extracted.shape[2]+extracted.shape[2],
            x[1]*extracted.shape[3]:x[1]*extracted.shape[3]+extracted.shape[3]  ] = extracted[x[0], x[1]]
    
    for i in range(extracted.shape[0]):
        for j in range(extracted.shape[1]):
            set_extract([i, j])
    return r

def draw_point(image, point, color=[0, 255, 0], radius=5):
    image = cv2.circle(image, center=(
                int(point[1]*image.shape[1]), 
                int(point[0]*image.shape[0])
            ), 
            radius=radius, color=color, thickness=-1)
    return image

def draw_coordinate_map(image, coordinate_map, color=[0, 255, 0], radius=5):
    temp = image.copy()
    for i in range(coordinate_map.shape[0]):
        for j in range(coordinate_map.shape[1]):
            temp = draw_point(temp, (
                coordinate_map[i][j][0], 
                coordinate_map[i][j][1]
            ), color, radius)
    return temp

if __name__ == '__main__':
    s = OIDClient('192.168.1.31', 33333)
    print(s.request_image_load('f4d07a53ade71fea')); # 1

    image = s.request_image_widthdraw('f4d07a53ade71fea')
    print(image.shape)
    
    extracted, cmap = sliding_window(image, 93, 200)
    print(extracted.shape)

    img1 = shifted_reconstruct(extracted)
    img2 = draw_coordinate_map(image, cmap)
    img3 = draw_point(image.copy(),  cmap[1, -1])
    img4 = draw_point(extracted[1, -1],  [0.5, 0.5])

    print(cmap[1, -1])

    while True:
        cv2.imshow('window', img1)
        cv2.imshow('window2', img2)
        cv2.imshow('window3', img3)
        cv2.imshow('window4', img4)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  
