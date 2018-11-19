import scan_csv
from LabelGenerator import label_image
from client import OIDClient
from FastSlidingWindow import *

ld = scan_csv.open_dicts()
train_image_codec = ld['train codec']
print(len(train_image_codec.keys()))
#print(train_image_codec[list(train_image_codec.keys())[0]])

client = OIDClient('192.168.1.31', 33333)
X, Y, image, cmap = label_image('f4d07a53ade71fea', train_image_codec, ld['class names'], client, 100, 200)
print(X.shape)
print(Y.shape)

for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        for c in range(Y.shape[2]):
            if Y[i, j, c][0] > -1.0:
                print([Y[i, j, c][1], Y[i, j, c][3]])
                image = draw_point(image, [Y[i, j, c][1], Y[i, j, c][3]])

while True:
    cv2.imshow('window1', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  