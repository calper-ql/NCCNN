import scan_csv
from LabelGenerator import *
from client import OIDClient
from FastSlidingWindow import *
from Util import *

ld = scan_csv.open_dicts()
train_image_codec = ld['train codec']
print(len(train_image_codec.keys()))
#print(train_image_codec[list(train_image_codec.keys())[0]])

client = OIDClient('192.168.1.31', 33333)
X, Y, image, cmap = label_image('f4d07a53ade71fea',
    train_image_codec, ld['class names'], client, 45, 50)
print(X.shape)
print(Y.shape)

img1 = draw_from_label(image, Y, cmap)
img2 = draw_from_raw_labels(image, train_image_codec['f4d07a53ade71fea'])

while True:
    cv2.imshow('window1', img1)
    cv2.imshow('window2', img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  
