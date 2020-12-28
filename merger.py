from __future__ import absolute_import, division, print_function

import sys
import glob


import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

#depth estimate modules
from overall import networks
from overall.layers import disp_to_depth
from overall.utils import download_model_if_doesnt_exist,extractor,det_write,img_saver,make_frame_dict

import time

import torch 
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, datasets

import numpy as np
import cv2 

from overall.util import *

import argparse
import os 
import os.path as osp

#yolo modules
from overall.darknet import Darknet
from overall.preprocess import prep_image, inp_to_image

import pandas as pd
import random 
import pickle as pkl
import itertools
from collections import Counter

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser(description='Pre-crash detection')
   
    parser.add_argument("--image_path", help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument('--depth_output_path', type=str,
                        help='path to a output depth map', default='depth')
    parser.add_argument('--trap_path', type=str,
                        help='path to a binary trapezium', default='trap_vid')
    parser.add_argument('--model_path', type=str,
                        help='path to models folder', default="./overall/models")
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"],
                        default="stereo_640x192")
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')


    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--pallete_path", help = "Path to pallete colours (pickle file)", default = './overall/pallete')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "./overall/cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "./overall/yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    
    return parser.parse_args()


diffval_dict = {}
key_lst = []
glob_frame_end = 0
glob_frame_start = 0
def vid_crash_detect(resultframe_dict_lst,wait_count_threshold = 8,d_i_threshold=25,ratio_threshold=2.2,consist_count_thresh=2,is_new = False):
    global vehicle_dict,diffval_dict,key_lst, glob_frame_end, glob_frame_start,jj, depth
    flag = 0
    k = 0
    frame_dict = frame_dict_lst[-1]
    if is_new:
        diffval_dict = {}
        key_lst = []
    for label,d_i in frame_dict.items():
        if d_i:
            if label not in key_lst:                
                diffval_dict[label] = {'consist_count':1,'wait_count':0,'start':glob_frame_start,'d_i':[d_i]}
                key_lst.append(label)
            else:
                if (d_i/diffval_dict[label]['d_i'][0] > ratio_threshold) and (d_i>d_i_threshold) and (diffval_dict[label]['consist_count']>=consist_count_thresh) and (len(diffval_dict[label]['d_i'])>len(frame_dict_lst)/2):
                    flag = 1
                    diffval_dict[label]['d_i'].append(d_i)
                    print("<<<< Apply Heavy Breakes now | Object: ",label," | Frame number: ",jj," >>>>")
                    dp = distance_dict['car']
                    v = abs(1/dp - 1/depth)/dt #- 30
                    depth = dp
                    # print("Relative Velocity: ",abs(1e7*(v*5)-30))
                    # exit()
                elif (d_i/diffval_dict[label]['d_i'][0] > ratio_threshold) and (d_i>d_i_threshold) and (len(diffval_dict[label]['d_i'])>len(frame_dict_lst)/2):
                    print("Preemptive alert | Object: ",label," | Frame number: ",jj)
                    diffval_dict[label]['d_i'].append(d_i)
                    diffval_dict[label]['consist_count'] += 1
                    dp = distance_dict['car']
                    v = abs(1/dp - 1/depth)/dt #- 30
                    depth = dp
                    # if 
                    # print("Relative Velocity: ",abs(1e7*(v*5)-30))
                elif (d_i/diffval_dict[label]['d_i'][0] > 1):
                    diffval_dict[label]['d_i'].append(d_i)
                else:
                    diffval_dict[label]['d_i'].append(None)
                    if diffval_dict[label]['wait_count'] > wait_count_threshold:
                        del diffval_dict[label]
                        key_lst.remove(label)
    
    for label in key_lst:
        try:
            diffval_dict[label]['wait_count'] = diffval_dict[label]['start'] + len(frame_dict_lst) - len(diffval_dict[label]['d_i']) + dict(Counter(diffval_dict[label]['d_i']))[None]
        except:
            diffval_dict[label]['wait_count'] = len(frame_dict_lst) - len(diffval_dict[label]['d_i'])
        if diffval_dict[label]['wait_count'] > wait_count_threshold:
            del diffval_dict[label]
            key_lst.remove(label)



if __name__ == '__main__':
    args = arg_parse()
    
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    download_model_if_doesnt_exist(args.model_path,args.model_name)
    model_path = os.path.join(args.model_path, args.model_name)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()
    if os.path.isfile(args.image_path):
        paths = [args.image_path]
    elif os.path.isdir(args.image_path):
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))
    
    scales = args.scales
    batch_size = 1
    num_frames = 5
    stride = num_frames
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    num_classes = 80
    classes = load_classes('./overall/data/coco.names') 
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32
    if CUDA:
        model.cuda()
    model.eval()
    if not os.path.exists(args.det):
        os.makedirs(args.det)
    imlist = paths
    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))],[True]*len(imlist),[args.trap_path]*len(imlist)))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list1 = torch.FloatTensor(im_dim_list).repeat(1,2)
    if CUDA:
        im_dim_list1 = im_dim_list1.cuda()
    leftover = 0
    if (len(im_dim_list) % batch_size):
        leftover = 1
    i = 0
    jj = 38
    result_lst = []
    out_lst = []
    frame_dict_lst = []  
    depth = 0.0001
    dt = 1/20.0
    
    with torch.no_grad():
        for idx, (image_path,batch,im_name) in enumerate(zip(paths,im_batches,imlist)):
            if idx == len(imlist)-1:
                break
            # try:
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)
            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im_pil = extractor(pil.fromarray(colormapped_im),trap_path=args.trap_path)
            # im_pil = pil.fromarray(colormapped_im)
            im = im_pil[:, :, ::-1].copy() 

            if CUDA:
                batch = batch.cuda()    
            prediction = model(Variable(batch), CUDA)
            prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            if type(prediction) == int:
                i += 1
                continue
            prediction[:,0] += i*batch_size
            i += 1
            if CUDA:
                torch.cuda.synchronize()
            if jj == len(imlist):
                break
            output = prediction
            im_dim_list = torch.index_select(im_dim_list1, 0, output[:,0].long())
            scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
            output[:,1:5] /= scaling_factor
            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
            colors = pkl.load(open(args.pallete_path, "rb"))
            orig_deps_copy = im.copy()
            result = list(map(lambda x: det_write(x, orig_deps_copy, classes,colors, args.trap_path), output))
            frame_dict, distance_dict = make_frame_dict(result)
            det_names = pd.Series(im_name).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))
            list(map(img_saver, det_names, result))
            if len(frame_dict_lst)<num_frames:
                frame_dict_lst.append(frame_dict)
                glob_frame_end = len(frame_dict_lst)
                vid_crash_detect(frame_dict_lst)
            else:
                glob_frame_start = jj
                glob_frame_end = len(frame_dict_lst)
                frame_dict_lst.append(frame_dict)
                frame_dict_lst = frame_dict_lst[stride:]
                vid_crash_detect(frame_dict_lst,is_new=True)    
            # except:
            #     continue
            jj += 1
            






    
