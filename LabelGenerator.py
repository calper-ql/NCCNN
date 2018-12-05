from FastSlidingWindow import *
from client import OIDClient

def label_image(image_id, codec, encodings, image_client, stride, size):
    if image_id not in codec:
        #print('Image id not in codec')
        return None, None, None, None
    resp = image_client.request_image_load(image_id)
    if resp != 'found':
        #print('Server does not have image')
        return None, None, None, None
    image = image_client.request_image_widthdraw(image_id)
    label_list = codec[image_id]

    index_encoding = {}
    count = 0
    for key in encodings:
        index_encoding[key] = count
        count += 1
    
    extracted, cmap = sliding_window(image, stride, size)
    cmap_orig = cmap.copy()
    cmap = np.reshape(cmap, [cmap.shape[0] * cmap.shape[1], 2])
    np_label = np.zeros([extracted.shape[0] * extracted.shape[1], count, 5])
    np_label[:, :, 0] = -1.0
    
    x_ratio = image.shape[1]/size
    y_ratio = image.shape[0]/size

    for label in label_list:
        x_size = float(label[4]) - float(label[3])
        x_center = float(label[3]) + x_size/2
        x_size = x_size/2

        y_size = float(label[6]) - float(label[5])
        y_center = float(label[5]) + y_size/2
        y_size = y_size/2

        for patch_id in range(np_label.shape[0]):
            if cmap[patch_id][1]+(size/2)/image.shape[1] >= y_center - y_size and cmap[patch_id][1]-(size/2)/image.shape[1] <= y_center + y_size:
                if cmap[patch_id][0]+(size/2)/image.shape[0] >= x_center - x_size and cmap[patch_id][0]-(size/2)/image.shape[0] <= x_center + x_size:
                    np_label[patch_id][index_encoding[label[1]]][0] = 1.0
                    np_label[patch_id][index_encoding[label[1]]][1] = (x_center - cmap[patch_id][0]) * x_ratio
                    np_label[patch_id][index_encoding[label[1]]][2] = x_size * x_ratio
                    np_label[patch_id][index_encoding[label[1]]][3] = (y_center - cmap[patch_id][1]) * y_ratio
                    np_label[patch_id][index_encoding[label[1]]][4] = y_size * y_ratio
    np_label = np.reshape(np_label, [extracted.shape[0], extracted.shape[1], count, 5])
    return extracted, np_label, image, cmap_orig
    