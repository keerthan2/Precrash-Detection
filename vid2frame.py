# Program To Read video 
# and Extract Frames 
import cv2 
import argparse
import os

# Function to extract frames 
def FrameCapture(path,out_path): 
	
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 

    # Used as counter variable 
    count = 0

    # checks whether frames were extracted 
    success = 1

    while success: 

        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
        if not success:
            break
        # Saves the frames with frame-count 
        image = cv2.resize(image,(640,360))
        cv2.imwrite(out_path+"/%d.jpg" % count, image) 

        count += 1

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser()
   
    parser.add_argument("--video_path", required=True ,type = str)
    parser.add_argument("--frame_path", required=True ,type = str)
    
    return parser.parse_args()

if __name__=='__main__':
    args = arg_parse()

    vid_path = args.video_path
    out_path = args.frame_path

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    FrameCapture(vid_path,out_path) 
