class_id_to_index = {'n02691156': 1, 'n02419796': 2, 'n02131653': 3, 'n02834778': 4, 'n01503061': 5, 'n02924116':
    6, 'n02958343': 7, 'n02402425': 8, 'n02084071': 9, 'n02121808': 10, 'n02503517': 11, 'n02118333': 12,
    'n02510455': 13, 'n02342885': 14, 'n02374451': 15, 'n02129165': 16, 'n01674464': 17,
    'n02484322': 18, 'n03790512': 19, 'n02324045': 20, 'n02509815': 21, 'n02411705': 22, 'n01726692': 23,
    'n02355227': 24, 'n02129604': 25, 'n04468005': 26, 'n01662784': 27, 'n04530566': 28, 'n02062744': 29, 'n02391049': 30}

import xml.etree.ElementTree as ET


'''
One annotation refers to one frame in ground truth. 
One label refers to one text file for one frame 
'''

def extract_from_xml(filename):
    tree = ET.parse(filename)  #xmlfile
    root = tree.getroot()
    # extract width and height
    width = int(root[3][0].text)
    height = int(root[3][1].text)
    # extract detections (tag with name equals object)
    annotation = []  # contains [id, x_min, y_min, x_max, y_max]
    for elem in root:
        if elem.tag == "object":
            xmax = elem[2][0].text
            xmin = elem[2][1].text
            ymax = elem[2][2].text
            ymin = elem[2][3].text
            annotation.append([class_id_to_index[elem[1].text], xmin, ymin, xmax, ymax])
    return annotation

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
                label_string += str(elem)
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


    # extract 100 images from 30 training sequence
    # VID_train_dir = '/Users/jimmyzhan/Documents/csc413/video_image_dataset/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000'
    # VID_train_ann_dir = '/Users/jimmyzhan/Documents/csc413/video_image_dataset/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000'
    # output_train_image_dir = '/Users/jimmyzhan/Documents/csc413/csc413-final-project/datasets/ImageNetVID/images/VID/train/'
    # output_train_label_dir = '/Users/jimmyzhan/Documents/csc413/csc413-final-project/datasets/ImageNetVID/labels/VID/train/'
    VID_val_dir = '/Users/jimmyzhan/Documents/csc413/video_image_dataset/ILSVRC2015/Data/VID/val'
    VID_val_ann_dir = '/Users/jimmyzhan/Documents/csc413/video_image_dataset/ILSVRC2015/Annotations/VID/val'
    output_val_image_dir = '/Users/jimmyzhan/Documents/csc413/csc413-final-project/test_sequences/images/'
    output_val_label_dir = '/Users/jimmyzhan/Documents/csc413/csc413-final-project/test_sequences/labels/'
    def add_zero(num, length=6):
        zero_length = length - len(str(num))
        return '0' * zero_length + str(num)
    # get hand_pick_seq
    hand_pick_seq = ['ILSVRC2015_val_00000000','ILSVRC2015_val_00000001', 'ILSVRC2015_val_00000002', 'ILSVRC2015_val_00002000', 'ILSVRC2015_val_00005001', 'ILSVRC2015_val_00007006', 'ILSVRC2015_val_00010000']
    hand_pick_seq_preferred_name = ['the-turtle', 'turtle-1-occlusion', 'turtle-2-occlusion', 'lizard-slow-move', 'multiple-zebras', 'jet-fast-move', 'zebra-occlusion']
    hand_pick_seq_images = [0, 364, 1, 91, 0, 143, 41]
    for i, seq in enumerate(hand_pick_seq): # e.g. ILSVRC2015_VID_val_00000000
        seq_path = os.path.join(VID_val_dir, seq)
        if os.path.isdir(seq_path):
            # create a seq folder for copied images
            image_dir = os.path.join(output_val_image_dir, hand_pick_seq_preferred_name[i])
            # print(image_dir)
            os.mkdir(image_dir)
            # create a seq folder for labels
            label_dir = os.path.join(output_val_label_dir, hand_pick_seq_preferred_name[i])
            # print(label_dir)
            os.mkdir(label_dir)
            for j in range(100): # 000000.JPEG
                image = add_zero(hand_pick_seq_images[i]+j) + ".JPEG" # 000001.JPEG
                seq_image = os.path.join(seq, image) # e.g. ILSVRC2015_val_00000000/000000.JPEG
                image_path = os.path.join(VID_val_dir, seq_image)
                # print(image_path)
                annotation_path = os.path.join(VID_val_ann_dir, seq_image)[:-4] + "xml"
                # print(annotation_path)
                image_dest = os.path.join(image_dir, image)
                # print(image_dest)
                shutil.copyfile(
                    image_path,
                    image_dest,
                    follow_symlinks=True)
                label_path = os.path.join(label_dir, image[:-4] + "txt")
                # print(label_path)
                write_label_to_txt(label_path, extract_from_xml(annotation_path))

    
    count_seq = 0
    for i, seq in enumerate(os.listdir(VID_val_dir)): # e.g. ILSVRC2015_VID_val_00000000
        if seq in hand_pick_seq: 
            continue
        if count_seq >= 23:
            break
        seq_path = os.path.join(VID_val_dir, seq)
        if os.path.isdir(seq_path):
            if len(os.listdir(seq_path)) < 100:
                continue
            count_seq += 1
            # create a seq folder for copied images
            image_dir = os.path.join(output_val_image_dir, seq)
            # print(image_dir)
            os.mkdir(image_dir)
            # create a seq folder for labels
            label_dir = os.path.join(output_val_label_dir, seq)
            # print(label_dir)
            os.mkdir(label_dir)
            count_image = 0 
            sorted_seq_dir = os.listdir(seq_path)
            sorted_seq_dir.sort()
            for image in sorted_seq_dir: # 000000.JPEG
                if count_image >= 100:
                    break
                seq_image = os.path.join(seq, image) # e.g. ILSVRC2015_val_00000000/000000.JPEG
                image_path = os.path.join(VID_val_dir, seq_image)
                # print(image_path)
                annotation_path = os.path.join(VID_val_ann_dir, seq_image)[:-4] + "xml"
                # print(annotation_path)
                image_dest = os.path.join(image_dir, image)
                # print(image_dest)
                shutil.copyfile(
                    image_path,
                    image_dest,
                    follow_symlinks=True)
                label_path = os.path.join(label_dir, image[:-4] + "txt")
                # print(label_path)
                write_label_to_txt(label_path, extract_from_xml(annotation_path))
                count_image += 1

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


    # # extract turtle validation first 100 images, annotations
    # print("here")
    # for i, file in enumerate(os.listdir("/Users/jimmyzhan/Documents/csc413/video_image_dataset/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00000000")):
    #     image = os.path.join("/Users/jimmyzhan/Documents/csc413/video_image_dataset/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00000000", file)
    #     print(image)
    #     end_part = file[:-4]
    #     image_dest = "/Users/jimmyzhan/Documents/csc413/csc413-final-project/turtle-100/images/" + "turtle-" + file 
    #     print(image_dest)
    #     shutil.copyfile(
    #             image,
    #             image_dest,
    #             follow_symlinks=True)
    #     image_text = "/Users/jimmyzhan/Documents/csc413/csc413-final-project/turtle-100/labels/" + "turtle-" + end_part + "txt"
    #     print(image_text)
    #     image_annotation = os.path.join('/Users/jimmyzhan/Documents/csc413/video_image_dataset/ILSVRC2015/Annotations/VID/val/ILSVRC2015_val_00000000', end_part + "xml")
    #     print(image_annotation)
    #     write_label_to_txt(image_text, annotation_to_label(image_annotation))
        
        

        






