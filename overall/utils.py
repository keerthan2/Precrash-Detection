from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib
import cv2
import numpy as np
import random 

def generate_trap(trap_path,height = 1000,width = 250,base = 50, start = 1):
    trap = np.zeros((2000,2000,3))
    for i in range(start,height):
      for j in range(999-base-int(i*(1000-width-base)/height)):
        trap[2000-i,1000-j,:] = 255
        trap[2000-i,1000+j,:] = 255
    cv2.imwrite(trap_path,trap)
    return 0
    
def extractor(img, trap_path,flag = 1):
    if flag == 0:
      generate_trap(trap_path,1000,250,50,50)
    shape = img.size
    trap = cv2.imread(trap_path)
    trap = cv2.resize(trap,(shape[0],shape[1]))
    trap = trap/255
    ex_img = np.multiply(trap,img)
    return ex_img

def det_write(x, res, classes,colors,trap_path):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    x1,y1 = c1
    x2,y2 = c2
    dist_dict = {}
    # img = res[int(x[0])]#.copy()
    img = res
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    if label == 'truck':
        label = 'car'
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    norm_dist = np.sum(img[y1:y2,x1:x2,:])#/np.sum(img)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    trap_binary = cv2.resize(cv2.imread(trap_path)/255.0,(img.shape[1],img.shape[0]))
    area_inside_trap = np.sum(trap_binary[y1:y2,x1:x2,:])
    dist_dict[label] = norm_dist*area_inside_trap
    return img,dist_dict,c2,norm_dist

def img_saver(names,result):
    cv2.imwrite(names,result[0])

def make_frame_dict(results):
    frame_dict = {}
    label_track = []
    dist_track = {}
    for i in range(len(results)):
        label = str([x for x in results[i][1].keys()][0])
        d_i = float([x for x in results[i][1].values()][0])
        ca = results[i][-2]
        cb = results[i][-1]
        if label == 'truck':
            label = 'car'
        if label in ['car','truck']:
            if label in label_track:
                if (d_i/1000000000)>frame_dict[label]:
                    frame_dict[label] = d_i/1000000000
                    dist_track[label] = results[i][-1]
            else:
                if d_i:
                    frame_dict[label] = d_i/1000000000
                    dist_track[label] = results[i][-1]
            

            # if label in label_track:
            #     if (d_i/100000)>frame_dict[label]:
            #         frame_dict[label] = d_i/100000
            # else:
            #     if d_i:
            #         frame_dict[label] = d_i/100000
    return frame_dict,dist_track

def make_frame_dict2(results):
    frame_dict = {}
    label_track = []
    for i in range(len(results)):
        label = str([x for x in results[i][1].keys()][0])
        d_i = float([x for x in results[i][1].values()][0])
        if label == 'truck':
            label = 'car'
        if label in ['car','bike','truck','motorcycle']:
            if label in label_track:
                if (d_i/1000000000)>frame_dict[label]:
                    frame_dict[label] = d_i/1000000000
            else:
                if d_i:
                    frame_dict[label] = d_i/1000000000
            # if label in label_track:
            #     if (d_i/100000)>frame_dict[label]:
            #         frame_dict[label] = d_i/100000
            # else:
            #     if d_i:
            #         frame_dict[label] = d_i/100000
    return frame_dict

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_path,model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_path = os.path.join(model_path, model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))
