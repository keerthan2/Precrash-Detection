# Precrash-Detection
Submission for Finals of Continental Fiction2Science 2019 Hackathon. <br>
Secured the highest score for ideation and implementation.

![example](figs/example.JPG)

# Usage
1. Run the `vid2frame.py` to convert video to frames. The below is an example of usage
```
    python vid2frame.py --video_path test_vid3.mp4 --frame_path test_vid3 
    [ This will read test_vid3.mp4 and save the frames in test_vid3 ]
```
2. Execute `merger.py` for precrash detection. The below is an example of usage
```
  python merger.py --image_path test_vid3 --det det --trap_path trap_vid3.png  
  [ This will read the frames from test_vid3 along with the given trapezium mask trap_vid3.png ]
```

3. Alert will be given as two types:
    * Preemptive alert: When there is suspicion of an accident, there will be an ouput in the terminal stating preemptive alert along with the object identified and frame number in which the alert was thrown.
    * Ultimate alert: This is when the algorithm knows for sure the accident is going to occur. In this case, it will output APPLY HEAVY BREAKS NOW along with the object identified and frame number in which the alert was thrown.
    * In case when there is no event of suspicion of crash, no output is given.

# Note
* The trapezium mask that has to be input (`trap_vid3.png` in the above examples) is specific for the dashboard dimension of the car.

