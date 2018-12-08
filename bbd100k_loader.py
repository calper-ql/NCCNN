import json
import cv2
from FastSlidingWindow import *
from Util import *
import math

class BBD100K_Loader:
    def __init__(self, is_train):
        self.is_train = is_train
        if self.is_train:
            with open('bbd100k/labels/bdd100k_labels_images_train.json') as f:
                self.raw_data = json.load(f)
        else:
            with open('bbd100k/labels/bdd100k_labels_images_val.json') as f:
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
        
    def gather(self, index, stride, size, overlap_ratio):
        if len(self.raw_data) <= index:
            return None, None, None, None
        image = None
        try:
            if self.is_train:
                image = cv2.imread('bbd100k/images/100k/train/' + self.raw_data[index]['name'])
            else:
                image = cv2.imread('bbd100k/images/100k/val/' + self.raw_data[index]['name'])
        except:
            return None, None, None, None

        index_encoding = {}
        count = 0
        for key in self.category_dict:
            index_encoding[key] = count
            count += 1
        
        extracted, cmap = sliding_window(image, stride, size)
        cmap_orig = cmap.copy()
        cmap = np.reshape(cmap, [cmap.shape[0] * cmap.shape[1], 2])
        np_label = np.zeros([extracted.shape[0] * extracted.shape[1], count, 5])

        o_r = overlap_ratio

        x_ratio = image.shape[1]/size
        y_ratio = image.shape[0]/size
        
        distances = np.ones([extracted.shape[0] * extracted.shape[1], count, 1]) * np.inf

        for label in self.raw_data[index]['labels']:
            if label['category'] != 'drivable area' and label['category'] != 'lane':
                bbox = label['box2d']
                x1 = float(bbox['x1'])
                y1 = float(bbox['y1'])
                x2 = float(bbox['x2'])
                y2 = float(bbox['y2'])

                x_size = float(x2) - float(x1)
                x_size /= image.shape[1]
                if x_size > 0.7:
                    return None, None, None, None
                x_center = float(x1)/image.shape[1] + x_size/2
                x_size = x_size/2

                y_size = float(y2) - float(y1)
                y_size /= image.shape[0]
                if y_size > 0.7:
                    return None, None, None, None
                y_center = float(y1)/image.shape[0] + y_size/2
                y_size = y_size/2

                for patch_id in range(np_label.shape[0]):
                    if cmap[patch_id][0]+(size/2)/image.shape[0] >= y_center - (y_size*o_r) and cmap[patch_id][0]-(size/2)/image.shape[0] <= y_center + (y_size*o_r):
                        if cmap[patch_id][1]+(size/2)/image.shape[1] >= x_center - (x_size*o_r) and cmap[patch_id][1]-(size/2)/image.shape[1] <= x_center + (x_size*o_r):
                            y_diff = (cmap[patch_id][0]+(size/2)/image.shape[0]) - y_center
                            x_diff = (cmap[patch_id][1]+(size/2)/image.shape[1]) - x_center
                            euc = x_diff*x_diff + y_diff*y_diff
                            if distances[patch_id][index_encoding[label['category']]][0] > euc:
                                distances[patch_id][index_encoding[label['category']]][0] = euc
                                np_label[patch_id][index_encoding[label['category']]][0] = 1.0
                                np_label[patch_id][index_encoding[label['category']]][1] = (x_center - cmap[patch_id][0]) * x_ratio
                                np_label[patch_id][index_encoding[label['category']]][2] = x_size * x_ratio
                                np_label[patch_id][index_encoding[label['category']]][3] = (y_center - cmap[patch_id][1]) * y_ratio
                                np_label[patch_id][index_encoding[label['category']]][4] = y_size * y_ratio
        np_label = np.reshape(np_label, [extracted.shape[0], extracted.shape[1], count, 5])
        return extracted, np_label, image, cmap_orig
        
    


if __name__ == "__main__":
    loader = BBD100K_Loader(True)
    color_map = generate_color_from_categories(loader.category_dict)
    for i in range(30, 31):
        X, Y, image, cmap = loader.gather(i, 50, 150, 0.1)
        print(X.shape)
        if image is not None:
            print(color_map)
            cv2.imshow('testY', draw_from_label(image, Y, cmap, 150, color_map, draw_patches=True))
            cv2.imshow('testP', draw_patches(image, cmap, 150))
            #cv2.imshow('test', draw_coordinate_map(image, cmap))
            cv2.waitKey()
    
