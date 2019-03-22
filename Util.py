import numpy as np
import cv2
import IPython
import random

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

def center_rectangle(image, center, sizes, color=[0, 255, 0], thickness=2):
    center = np.array(center)
    sizes = np.array(sizes)
    p1 = [(center[0]-sizes[0]) * image.shape[1], (center[1]-sizes[1]) * image.shape[0]]
    p2 = [(center[0]+sizes[0]) * image.shape[1], (center[1]+sizes[1]) * image.shape[0]]
    image = cv2.rectangle(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color=color, thickness=thickness)
    return image

def generate_color_from_categories(categories, scale=False):
    generated_colors = {}
    inc = 255.0/(len(categories.keys()))
    for i in range(len(categories.keys())):
        hsv = np.uint8([[[i*inc, 200+random.uniform(0, 1.0)*55, 150+random.uniform(0, 1.0)*105]]])
        col = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        generated_colors[list(categories.keys())[i]] = [int(col[0]), int(col[1]), int(col[2])]#[random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)]
    if scale:
        for m in generated_colors:
            generated_colors[m] = np.array(generated_colors[m])/255.0
    return generated_colors

def draw_from_label(image, Y, color_map, rie, treshold=0.9, meta_size=4, print_labels=True):
    encode_length = len(rie) + meta_size
    el = encode_length
    pw = (image.shape[1]/(Y.shape[1])) / image.shape[1]
    ph = (image.shape[0]/(Y.shape[0])) / image.shape[0]
    for i in range(Y.shape[1]):
        for j in range(Y.shape[0]):
            for k in range(Y.shape[2]//encode_length):
                data = Y[j, i, k*el:(k+1)*el]
                classes = data[0:len(color_map)]
                best = np.argmax(classes)
                if classes[best] > treshold:
                    image = center_rectangle(image, 
                                     [data[len(rie)+0]+pw*i, data[len(rie)+1]+ph*j], 
                                     [data[len(rie)+2]/2.0, data[len(rie)+3]/2.0], 
                                     color_map[rie[best]])
                    if print_labels:
                        text = rie[best].upper()
                        text_x = int( (data[len(rie)+0]+pw*i - data[len(rie)+2]/2.0) * image.shape[1])
                        text_y = int( (data[len(rie)+1]+ph*j - data[len(rie)+3]/2.0) * image.shape[0])
                        
                        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, thickness=1)[0]
                        box_coords = ((text_x, text_y), (text_x + text_width - 2, text_y - text_height-4))
                        cv2.rectangle(image, box_coords[0], box_coords[1], color_map[rie[best]], cv2.FILLED)
                        image = cv2.putText(image, text, (text_x, text_y-2), cv2.FONT_HERSHEY_DUPLEX, 0.6, [0, 0, 0], 1, lineType=cv2.LINE_AA)
    return image

