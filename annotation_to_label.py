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
    import shutil

#     # # experiment copy file
#     # shutil.copyfile('/Users/jimmyzhan/Documents/csc413/csc413-final-project/datasets/ImageNetVID/labels/VID/train/a.html',
#     #                 '/Users/jimmyzhan/Documents/csc413/csc413-final-project/datasets/ImageNetVID/labels/VID/train/b.html', follow_symlinks=True)


#     # extract one image from each training sequence
#     VID_val_dir = '/Users/jimmyzhan/Documents/csc413/video_image_dataset/ILSVRC2015/Data/VID/val'
#     VID_train_dir = '/Users/jimmyzhan/Documents/csc413/video_image_dataset/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000'
#     VID_train_ann_dir = '/Users/jimmyzhan/Documents/csc413/video_image_dataset/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000'
#     VID_val_ann_dir = '/Users/jimmyzhan/Documents/csc413/video_image_dataset/ILSVRC2015/Annotations/VID/val'
#     output_train_image_dir = '/Users/jimmyzhan/Documents/csc413/csc413-final-project/datasets/ImageNetVID/images/VID/train/'
#     output_val_image_dir = '/Users/jimmyzhan/Documents/csc413/csc413-final-project/datasets/ImageNetVID/images/VID/val/'
#     output_train_label_dir = '/Users/jimmyzhan/Documents/csc413/csc413-final-project/datasets/ImageNetVID/labels/VID/train/'
#     output_val_label_dir = '/Users/jimmyzhan/Documents/csc413/csc413-final-project/datasets/ImageNetVID/labels/VID/val/'

#     for i, filename in enumerate(os.listdir(VID_train_dir)): # e.g. ILSVRC2015_VID_train_00000000
#         f = os.path.join(VID_train_dir, filename)
#         if os.path.isdir(f):
#             end_part_image_0_name = filename + '-' + os.listdir(f)[0]  # e.g. ILSVRC2015_train_00000000-000000.JPEG
#             end_part_image_0 = os.path.join(filename, os.listdir(f)[0]) # e.g. ILSVRC2015_train_00000000/000000.JPEG
#             image_0 = os.path.join(VID_train_dir, end_part_image_0)
#             # print(image_0)
#             image_0_annotation = os.path.join(VID_train_ann_dir, end_part_image_0)[:-4] + "xml"
#             # print(image_0_annotation)
#             image_0_dest = output_train_image_dir + end_part_image_0_name
#             # print(image_0_dest)
#             shutil.copyfile(
#                 image_0,
#                 image_0_dest,
#                 follow_symlinks=True)
#             image_0_text = output_train_label_dir + end_part_image_0_name[:-4] + "txt"
#             # print(image_0_text)
#             write_label_to_txt(image_0_text, annotation_to_label(image_0_annotation))

    # # train image path
    # '/Users/jimmyzhan/Documents/csc413/video_image_dataset/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000/000000.JPEG'
    # # train annotation path
    # '/Users/jimmyzhan/Documents/csc413/video_image_dataset/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000/000000.xml'
    # # new train image path
    # '/Users/jimmyzhan/Documents/csc413/csc413-final-project/datasets/ImageNetVID/images/VID/train'
    # # train label path
    # '/Users/jimmyzhan/Documents/csc413/csc413-final-project/datasets/ImageNetVID/labels/VID/train'
    #
    #
    #
    # # convert xml annotation to txt label
    # for xml_filename in os.listdir('/Users/jimmyzhan/Documents/csc413/video_image_dataset/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00139005/000000.JPEG'):
    #     f = os.path.join('/Users/jimmyzhan/Documents/csc413/csc413-final-project/datasets/ImageNetVID/labels/VID/val', xml_filename)
    #     if os.path.isfile(f):
    #         if f.endswith('.xml'):
    #             write_label_to_txt(f[:-3] + "txt", annotation_to_label(f))


    # extract turtle validation first 100 images, annotations
    print("here")
    for i, file in enumerate(os.listdir("/Users/jimmyzhan/Documents/csc413/video_image_dataset/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00000000")):
        image = os.path.join("/Users/jimmyzhan/Documents/csc413/video_image_dataset/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00000000", file)
        print(image)
        end_part = file[:-4]
        image_dest = "/Users/jimmyzhan/Documents/csc413/csc413-final-project/turtle-100/images/" + "turtle-" + file 
        print(image_dest)
        shutil.copyfile(
                image,
                image_dest,
                follow_symlinks=True)
        image_text = "/Users/jimmyzhan/Documents/csc413/csc413-final-project/turtle-100/labels/" + "turtle-" + end_part + "txt"
        print(image_text)
        image_annotation = os.path.join('/Users/jimmyzhan/Documents/csc413/video_image_dataset/ILSVRC2015/Annotations/VID/val/ILSVRC2015_val_00000000', end_part + "xml")
        print(image_annotation)
        write_label_to_txt(image_text, annotation_to_label(image_annotation))
        
        

        






