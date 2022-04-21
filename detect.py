# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xywhn2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from seq_nms import seq_nms
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from more_itertools import sort_together
import glob
import numpy as np


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        # source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        test_seqs=ROOT / 'test_sequences',  # dir for the test set containing images and ground truth labels
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        use_seq_nms=False,  # use seq_nms
        use_modified_seq_nms=False  # use modified seq_nms
):
    apply_original_nms = True
    if use_seq_nms or use_modified_seq_nms:
        apply_original_nms = False

    source = str(test_seqs)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # save_img = False  ## debugging mode
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

    # source dir contains img dirs, each img dir has frames from 1 video
    seq_i = 0
    ap_sum = 0
    for img_folder in sorted(glob.glob(source + '/images/*')):  # run inference on each sequence
        dest_dir_name = img_folder.split('/')[-1]
        # Directories
        save_dir = Path(project) / (str(seq_i) + "_" + dest_dir_name)
        print(save_dir)
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Dataloader
        if webcam:  # not used
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(img_folder, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size  ## Can modify batch_size or add as command line argument
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        dt, seen = [0.0, 0.0, 0.0], 0

        ## tensor for storing all bbox detections for all images in the sequence
        ## this is the input tensor to seq_nms()
        ## len(dataset) == num_frames
        seq_bboxes = []  ## make into tensor of shape (num_frames, num_boxes, 4)
        seq_scores = []  ## make into tensor of shape (num_frames, num_boxes)
        max_num_bboxes = 0

        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, apply_original_nms=apply_original_nms)
            dt[2] += time_sync() - t3

            ## add detected bboxes to lists
            bboxes = pred[0][..., :4]
            scores = pred[0][..., 4]
            seq_bboxes.append(bboxes)
            seq_scores.append(scores)
            max_num_bboxes = max(max_num_bboxes, bboxes.shape[0])

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            ## Can save a different version into txt
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        ## pad seq_bboxes and seq_scores with zeros by difference between max_num_bboxes and current seq_bboxes's frame
        # len(seq_bboxes)==len(seq_scores)==len(dataset)==the number of image frames in the sequence
        padded_bboxes = torch.zeros(len(seq_bboxes), max_num_bboxes, 4)
        padded_scores = torch.zeros(len(seq_bboxes), max_num_bboxes)
        for i in range(len(seq_bboxes)): # each frame
            num_bboxes_this_frame = seq_bboxes[i].shape[0]
            padded_scores[i][:num_bboxes_this_frame] = seq_scores[i]
            for j in range(seq_bboxes[i].shape[0]): # each bbox in the frame
                padded_bboxes[i][j] = seq_bboxes[i][j]

        if use_seq_nms or use_modified_seq_nms:
            t_1 = time_sync()
            best_pred_bboxes_seq = seq_nms(padded_bboxes, padded_scores, use_modified_seq_nms=use_modified_seq_nms)
            t_2 = time_sync()
            print(f'{t_2 - t_1:.3f}s for seq_nms')

            seq_nms_dir_name = "dir"
            if use_seq_nms:
                seq_nms_dir_name = "seq_nms"
            elif use_modified_seq_nms:
                seq_nms_dir_name = "modified_seq_nms"

            # turn into a dictionary of key as frame_idx, and value as list of tuples (x1,y1,x2,y2,score)
            pred_bboxes_seq = {}
            ## and save best_seqs into file
            with open(str(save_dir) + seq_nms_dir_name + '_results.txt', 'a') as f:
                for frame_idx, seq in best_pred_bboxes_seq.items():  # key value pair
                    pred_bboxes_seq[frame_idx] = []
                    for bbox, score in seq:
                        pred_bboxes_seq[frame_idx].append((bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item(), score.item()))
                        f.write(f'{frame_idx} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {score} ')  ## check for multi-bbox frame
                        f.write('\n')
            
            ## save visualizations for (modified) seq-nms processed images

            (save_dir / seq_nms_dir_name).mkdir(parents=True, exist_ok=True)  # make viz folder
            
            for f_idx, img_x in enumerate(dataset):
                path, im, im0s, vid_cap, s = img_x
                annotator = Annotator(im0s, line_width=line_thickness, example=str(names))
                if f_idx in pred_bboxes_seq:
                    for v_bbox in pred_bboxes_seq[f_idx]:
                        annotator.box_label(v_bbox[:4], label=str(v_bbox[4]), color=colors(c, True))
                img_path_output = str(save_dir / seq_nms_dir_name) + "/_" + str(f_idx) + ".JPEG"
                # print(img_path_output)
                cv2.imwrite(img_path_output, annotator.result())

        else:
            # seq_bboxes = [tensor([[420.,   5., 613., 187.], 
            #                       [...]]), 
            #               tensor([[421.,  13., 614., 197.]]), ... ]
            # seq_scores = [tensor([0.38223]), tensor([0.47908]), ...]
            pred_bboxes_seq = {}
            with open(str(save_dir / 'nms_results.txt'), 'a') as f:
                for frame_idx, frame in enumerate(seq_bboxes):
                    pred_bboxes_seq[frame_idx] = []
                    for i, bbox in enumerate(seq_bboxes[frame_idx]):
                        b_score = seq_scores[frame_idx][i].item()
                        pred_bboxes_seq[frame_idx].append((bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item(), b_score))
                        f.write(f'{frame_idx} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {b_score} ')  ## check for multi-bbox frame  

        # pred_bboxes_seq =
        # {0: [(420.0, 5.0, 613.0, 187.0, 0.5108696222305298), (...)], 
        #  1: [(421.0, 13.0, 614.0, 197.0, 0.5108696222305298)],

        gt_bboxes_seq = {}
        num_gt = 0
        gt_dir = sorted(glob.glob(test_seqs + "/labels/*"))[seq_i]
        for f_i, gt_file in enumerate(sorted(glob.glob(gt_dir + "/*.txt"))):
            # print("gt labels filepath:", gt_file)
            gt_bboxes_seq[f_i] = []
            with open(gt_file, "r") as f:
                while True:  # while not at bottom of file
                    # read line
                    line = f.readline()  # represents one frame
                    # e.g. line = "1 0.530078113079071 0.4000000059604645 0.08359374850988388 0.06666667014360428 "
                    if not line:  # if line is empty (reaching end of file), break
                        f.close()
                        break
                    # split line into list
                    line = line.split(' ')
                    f_cls = int(line[0])
                    # bbox_tensor_xywh = torch.tensor([[float(line[1]),float(line[2]),float(line[3]),float(line[4])]])
                    # f_bbox = xywhn2xyxy(bbox_tensor_xywh).squeeze()
                    # gt_bboxes_seq[f_i].append((f_bbox[0].item(), f_bbox[1].item(), f_bbox[2].item(), f_bbox[3].item(), f_cls))
                    gt_bboxes_seq[f_i].append((float(line[1]),float(line[2]),float(line[3]),float(line[4]), f_cls))
                    num_gt += 1
        
        # print("pred_bboxes_seq: ", pred_bboxes_seq)
        # print("gt_bboxes_seq: ", gt_bboxes_seq)

        ## evaluate performance

        # get the true positives and corresponding pred conf scores
        is_tp_lst = []
        scores = []
        for frame_idx in pred_bboxes_seq:
            # if IoU of predicted bbox and gt bbox is greater than threshold, then it is a true positive
            pred_bboxes_frame = pred_bboxes_seq[frame_idx]
            # num_pred += len(pred_bboxes_frame)
            gt_bboxes_used = []
            for pred_bbox in pred_bboxes_frame:
                tensor_pred_bbox = torch.tensor(pred_bbox[:-1]).unsqueeze(0)
                is_tp = False
                for gt_bbox in gt_bboxes_seq[frame_idx]:
                    tensor_gt_bbox = torch.tensor(gt_bbox[:-1]).unsqueeze(0)
                    iou_pred_gt = box_iou(tensor_pred_bbox, tensor_gt_bbox).squeeze().item()
                    # print("iou_pred_gt: ", iou_pred_gt)
                    if iou_pred_gt > 0.85 and (gt_bbox not in gt_bboxes_used):  # TODO: try different thresholds
                        # num_tp += 1
                        gt_bboxes_used.append(gt_bbox)
                        is_tp = True
                        break
                if is_tp:
                    is_tp_lst.append(1)
                else:
                    is_tp_lst.append(0)
                scores.append(pred_bbox[-1])
        
        # calculate precision & recall
        if len(scores) > 0:
            scores, is_tp_lst = sort_together([scores, is_tp_lst])
        # print("scores", scores)
        # print("is_tp_lst", is_tp_lst)

        precision_lst = []
        recall_lst = []
        for i in range(len(is_tp_lst)):
            num_tp = sum(is_tp_lst[:i+1])
            num_detections = i+1
            precision_lst.append(num_tp / num_detections)
            recall_lst.append(num_tp / num_gt)
        
        print("precision_lst: ", precision_lst)
        print("recall_lst: ", recall_lst)

        # plot pr curve
        plt.plot(recall_lst, precision_lst)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(str(save_dir / 'PR_curve.png'))  ## specify to dest_folder

        # calculate the average precision (area under PR curve)
        if len(is_tp_lst) == 0:
            ap = 0
        else:
            ap = average_precision_score(is_tp_lst, scores)
            if not np.isnan(ap):
                ap_sum += ap
        print("AP of this sequence", dest_dir_name ,":", ap)

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
        
        seq_i += 1
    
    print("number of total test sequences:", seq_i)
    mean_ap = ap_sum / seq_i
    print("mAP:", mean_ap)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    # parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--use_seq_nms', action='store_true', help='apply seq-nms')
    parser.add_argument('--use_modified_seq_nms', action='store_true', help='apply modified seq-nms')
    parser.add_argument('--test_seqs', type=str, default=ROOT / 'test_sequences', help='dir for the test set containing images and ground truth labels')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
