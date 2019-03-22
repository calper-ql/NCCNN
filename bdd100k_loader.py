import json
import cv2
from Util import *
import math
from decimal import Decimal

class BDD100K_Loader:
    def __init__(self, is_train):
        self.is_train = is_train
        if self.is_train:
            with open('bdd100k/labels/bdd100k_labels_images_train.json') as f:
                self.raw_data = json.load(f)
        else:
            with open('bdd100k/labels/bdd100k_labels_images_val.json') as f:
                self.raw_data = json.load(f)
        self.image_names = []
        self.category_dict = {}
        for item in self.raw_data:
            self.image_names.append(item['name'])
            for label in item['labels']:
                if label['category'] in self.category_dict:
                    self.category_dict[label['category']] += 1
                else:
                    self.category_dict[label['category']] = 1
        del self.category_dict['drivable area']
        del self.category_dict['lane']
        self.image_count = len(self.image_names)
        self.indices = None
        print(self.category_dict)
        index_encoding = {}
        reverse_index_encoding = {}
        count = 0
        for key in self.category_dict:
            index_encoding[key] = count
            reverse_index_encoding[count] = key
            count += 1
        self.index_encoding = index_encoding
        self.reverse_index_encoding = reverse_index_encoding
        print(self.index_encoding)
        self.size = len(self.raw_data)
        self.encode_length = len(index_encoding) + 2 + 2
        
    def gather(self, index, region_size, encode_depth):
        if len(self.raw_data) <= index:
            return None, None
        image = None
        try:
            if self.is_train:
                image = cv2.imread('bdd100k/images/100k/train/' + self.raw_data[index]['name'])
            else:
                image = cv2.imread('bdd100k/images/100k/val/' + self.raw_data[index]['name'])
        except:
            return None, None

        ie = self.index_encoding
        
        encode_length = len(ie) + 2 + 2
        pw = Decimal(1.0/region_size[1])
        ph = Decimal(1.0/region_size[0])
        
        grand_data = np.zeros([region_size[0], region_size[1], encode_length*encode_depth])
        
        p_data = []
        for i in range(region_size[1]):
            p_data.append([])
            for j in range(region_size[0]):
                p_data[i].append([])
        
        for label in self.raw_data[index]['labels']:
            if label['category'] != 'drivable area' and label['category'] != 'lane':
                bbox = label['box2d']
                x1 = float(bbox['x1']) / image.shape[1]
                y1 = float(bbox['y1']) / image.shape[0]
                x2 = float(bbox['x2']) / image.shape[1]
                y2 = float(bbox['y2']) / image.shape[0]
                centerx = Decimal((x1 + x2) / 2.0)
                centery = Decimal((y1 + y2) / 2.0)
                tx = centerx % (pw)
                ty = centery % (ph)
                tw = x2 - x1
                th = y2 - y1 
                data = np.zeros([encode_length])
                data[ie[label['category']]] = 1.0
                data[len(ie)+0] = tx
                data[len(ie)+1] = ty
                data[len(ie)+2] = tw
                data[len(ie)+3] = th
                wi = int((centerx)//pw)
                hi = int((centery)//ph)
                p_data[wi][hi].append(data)
                    
        for i in range(region_size[1]):
            for j in range(region_size[0]):
                #print(i, j, len(p_data[i][j]))
                depth = p_data[i][j]
                depth = sorted(depth, key=lambda x: x[len(ie)+2]* x[len(ie)+3], reverse=True)
                for k in range(len(depth)):
                    x = depth[k]
                    if k < encode_depth:
                        grand_data[j, i, k*encode_length:(k+1)*encode_length] = x
                
        return image, grand_data

    
if __name__ == "__main__":
    print('Train')
    loader = BDD100K_Loader(True)
    color_map = generate_color_from_categories(loader.category_dict)
    rie = loader.reverse_index_encoding
    for i in range(30, 55):
        image, Y = loader.gather(i, [17, 23], 5)
        if image is not None:
            image = draw_from_label(image, Y, color_map, rie)
            cv2.imshow('image', image)
            cv2.waitKey()
            
    print('Validation')
    loader = BDD100K_Loader(False)
    color_map = generate_color_from_categories(loader.category_dict)
    rie = loader.reverse_index_encoding
    for i in range(30, 55):
        image, Y = loader.gather(i, [11, 33], 5)
        if image is not None:
            image = draw_from_label(image, Y, color_map, rie)
            cv2.imshow('image', image)
            cv2.waitKey()
    
