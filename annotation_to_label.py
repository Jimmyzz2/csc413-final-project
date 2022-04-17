class_id_to_index = {'n02691156': 1, 'n02419796': 2, 'n02131653': 3, 'n02834778': 4, 'n01503061': 5, 'n02924116':
    6, 'n02958343': 7, 'n02402425': 8, 'n02084071': 9, 'n02121808': 10, 'n02503517': 11, 'n02118333': 12,
    'n02510455': 13, 'n02342885': 14, 'n02374451': 15, 'n02129165': 16, 'n01674464': 17,
    'n02484322': 18, 'n03790512': 19, 'n02324045': 20, 'n02509815': 21, 'n02411705': 22, 'n01726692': 23,
    'n02355227': 24, 'n02129604': 25, 'n04468005': 26, 'n01662784': 27, 'n04530566': 28, 'n02062744': 29, 'n02391049': 30}

import xml.etree.ElementTree as ET
import torch
import os

'''
One annotation refers to one frame in ground truth. 
One label refers to one text file for one frame 
'''

def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() # assume input is torch.tensor
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2

def extract_from_annotation(filename):
    tree = ET.parse(filename)  #xmlfile
    root = tree.getroot()
    # extract width and height
    width = int(root[3][0].text)
    height = int(root[3][1].text)
    # extract detections (tag with name equals object)
    # object
    for elem in root:
        if elem.tag == "object":
            print(elem.tag)
    id = root[4][1].text
    xmax = int(root[4][2][0].text)
    xmin = int(root[4][2][1].text)
    ymax = int(root[4][2][2].text)
    ymin = int(root[4][2][3].text)
    width = int(root[3][0].text)
    height = int(root[3][1].text)
    return [xmin, ymin, xmax, ymax], [id, width, height]

def annotation_to_label(filenames):
    # for convert a sequence of annotations to labels
    n = len(filenames)
    annotations = torch.zeros((n, 4))  # nx4 with [x1, y1, x2, y2] unnormalized
    indexs = torch.zeros((n, 1))  # nx1 with [id]
    idwh = [0, 0, 0]  # [id, width, height]
    for i, filename in enumerate(filenames): # loop over xml files, each xml file contains 1 to few bboxes
        xyxy, idwh = extract_from_annotation(filename)
        annotations[i] = torch.tensor(xyxy)
        index = class_id_to_index[idwh[0]]
        indexs[i] = index
    label = xyxy2xywhn(annotations, idwh[1], idwh[2]) #nx4 with [x, y, w, h] x center, y center, width, height normalized
    label = torch.cat([indexs, label], dim=1)
    return label

def write_labels_to_txt(filename, label):
    '''
    :param filename: path to text output label
    :param label: label tensor, nx4 with [x, y, w, h]
    :return: None
    '''
    with open(filename, 'w') as f:
        for bbox in label:
            label_string = ""
            for i, elem in enumerate(bbox): # [index, x, y, w, h]
                if i == 0:  # elem = index
                    label_string += str(int(elem.item())) # convert index to int
                else:
                    label_string += str(elem.item())
                label_string += " "
            label_string += "\n"
            f.write(label_string)
        f.close()

# def path_


if __name__ == "__main__":
    xml_filenames = []
    for xml_filename in os.listdir('/Users/jimmyzhan/Documents/csc413/csc413-final-project/ILSVRC2015/Annotations/VID/train'):
        f = os.path.join('/Users/jimmyzhan/Documents/csc413/csc413-final-project/ILSVRC2015/Annotations/VID/train', xml_filename)
        if os.path.isfile(f):
            if f.endswith('.xml'):
                xml_filenames.append(f)
    write_labels_to_txt('/Users/jimmyzhan/Documents/csc413/csc413-final-project/ILSVRC2015/Annotations/VID/train/label.txt',
                                            annotation_to_label(xml_filenames))




