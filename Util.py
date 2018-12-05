import numpy as np
import cv2
import IPython

def imshow(img):
    from PIL import Image
    IPython.display.display(Image.fromarray(img))
    
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

def center_rectangle(image, center, sizes, color=[0, 255, 0], thickness=2):
    center = np.array(center)
    sizes = np.array(sizes)
    p1 = [(center[0]-sizes[0]) * image.shape[1], (center[1]-sizes[1]) * image.shape[0]]
    p2 = [(center[0]+sizes[0]) * image.shape[1], (center[1]+sizes[1]) * image.shape[0]]
    image = cv2.rectangle(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color=color, thickness=thickness)
    return image

def draw_from_label(image, label, cmap, size, confidence_treshold=0.0, draw_patches=False, thickness=1):
    image = image.copy()
    Y = label.copy()
    x_ratio = size/image.shape[1]
    y_ratio = size/image.shape[0]
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            for c in range(Y.shape[2]):
                if Y[i, j, c][0] > confidence_treshold:
                    if draw_patches:
                        image = center_rectangle(image,
                        [cmap[i, j, 0], 
                        cmap[i, j, 1]],
                        [size/(2.0*image.shape[1]), 
                        size/(2.0*image.shape[0])], color=[0, 0, 255], thickness=thickness)
                    image = center_rectangle(image,
                        [cmap[i, j, 0] + Y[i, j, c][1] * x_ratio, 
                        cmap[i, j, 1] +  Y[i, j, c][3] * y_ratio],
                        [Y[i, j, c][2] * x_ratio, 
                        Y[i, j, c][4] * y_ratio], thickness=thickness)
    return image

def draw_from_raw_labels(image, label_list):
    image = image.copy()
    for label in label_list:
        x_size = float(label[4]) - float(label[3])
        x_center = float(label[3]) + x_size/2
        x_size = x_size/2

        y_size = float(label[6]) - float(label[5])
        y_center = float(label[5]) + y_size/2
        y_size = y_size/2

        image = center_rectangle(image,
                    [x_center, 
                    y_center],
                    [x_size, 
                    y_size])
    return image