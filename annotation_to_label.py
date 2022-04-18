class_id_to_index = {'n02691156': 1, 'n02419796': 2, 'n02131653': 3, 'n02834778': 4, 'n01503061': 5, 'n02924116':
    6, 'n02958343': 7, 'n02402425': 8, 'n02084071': 9, 'n02121808': 10, 'n02503517': 11, 'n02118333': 12,
    'n02510455': 13, 'n02342885': 14, 'n02374451': 15, 'n02129165': 16, 'n01674464': 17,
    'n02484322': 18, 'n03790512': 19, 'n02324045': 20, 'n02509815': 21, 'n02411705': 22, 'n01726692': 23,
    'n02355227': 24, 'n02129604': 25, 'n04468005': 26, 'n01662784': 27, 'n04530566': 28, 'n02062744': 29, 'n02391049': 30}

import xml.etree.ElementTree as ET
import torch


'''
One annotation refers to one frame in ground truth. 
One label refers to one text file for one frame 
'''

def xyxy2xywhn(x, w=640, h=640):
    # Convert x1, y1, x2, y2 to x, y, w, h normalized where xy1=top-left, xy2=bottom-right
    y = x.clone()  # assume input is torch.tensor
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y

def extract_from_xml(filename):
    tree = ET.parse(filename)  #xmlfile
    root = tree.getroot()
    # extract width and height
    width = int(root[3][0].text)
    height = int(root[3][1].text)
    # extract detections (tag with name equals object)
    annotation = []  # contains [x1, y1, x2, y2] unnormalized
    index = []  # contains id
    for elem in root:
        if elem.tag == "object":
            index.append([int(class_id_to_index[elem[1].text])])
            xmax = float(elem[2][0].text)
            xmin = float(elem[2][1].text)
            ymax = float(elem[2][2].text)
            ymin = float(elem[2][3].text)
            annotation.append([xmin, ymin, xmax, ymax])
    return [torch.tensor(annotation), torch.tensor(index), width, height]

def annotation_to_label(filename):
    # convert annotation to label
    annotation, index, width, height = extract_from_xml(filename)
    label = xyxy2xywhn(annotation, width, height)  # num_detections x4 with [index, x, y, w, h]
    label = torch.cat([index, label], dim=1)
    return label

def write_label_to_txt(filename, label):
    '''
    :param filename: path to text output label
    :param label: label tensor, nx4 with [x, y, w, h]
    :return: None
    '''
    with open(filename, 'w') as f:
        for bbox in label:
            label_string = ""
            for i, elem in enumerate(bbox):  # [index, x, y, w, h]
                if i == 0:  # elem = index
                    label_string += str(int(elem.item()))  # convert index to int
                else:
                    label_string += str(elem.item())
                label_string += " "
            label_string += "\n"
            f.write(label_string)
        f.close()


if __name__ == "__main__":
    import os
    for xml_filename in os.listdir(
            '/Users/jimmyzhan/Documents/csc413/csc413-final-project/datasets/ImageNetVID/labels/VID/val'):
        f = os.path.join('/Users/jimmyzhan/Documents/csc413/csc413-final-project/datasets/ImageNetVID/labels/VID/val',
                         xml_filename)
        if os.path.isfile(f):
            if f.endswith('.xml'):
                write_label_to_txt(f[:-3] + "txt", annotation_to_label(f))





