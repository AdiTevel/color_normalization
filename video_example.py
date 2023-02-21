import cv2
from color_class import ColorClassificaton
from utils.semantic_segmentation_utils import SemanticSegmentationWrapper
import numpy as np
from nadav_color_classiffy import ColorClassifier as ncc
video_file  = '/home/nadav/Desktop/ambrosia_rgb/vid_220918_074339_C.avi'
init_frame  = 1000
cap = cv2.VideoCapture(video_file)
cap.set(1, init_frame-1)
color_class = ColorClassificaton()
old_color_class = ncc(True)
segment_object =  SemanticSegmentationWrapper()
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame
        key = cv2.waitKey() & 0xFF
        # Press Q on keyboard to  exit
        if key == ord('q'):
            break
        elif key == ord('c'):
            roi = cv2.selectROI(frame)
            x = int(roi[0] + roi[2] / 2)
            y = int(roi[1] + roi[3] / 2)
            r = int(max(roi[2] / 2, roi[3] / 2))
            out, box_out = segment_object.infer(frame,[[x,y,r]] , True)
            classes_segmented = color_class.infer_masked_pixles(frame, box_out)
            crop = frame[y-r:y+r,x-r:x+r,:]
            old_class = old_color_class.grade(crop,out[0])
            arr8 = (out[0]*1).astype(np.uint8)
            mask_colors = cv2.applyColorMap(arr8 * 100, cv2.COLORMAP_HSV)
            color_out = color_class.infer(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), xywh=[list(roi)])
            # frame[y - r:y + r, x - r:x + r, :] = mask_colors
            cv2.imshow('mask',mask_colors)
            cv2.waitKey(1)
            cv2.rectangle(frame,(roi[0],roi[1]),(roi[0]+roi[2],roi[1]+roi[3]),(255,0,0),2)
            cv2.putText(frame, f'{str(color_out[0][0]):.3},{str(old_class):.3}', (roi[0], roi[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.imshow('Frame', frame)


    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()