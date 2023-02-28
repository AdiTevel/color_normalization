import cv2
from color_class import ColorClassificaton
from utils.semantic_segmentation_utils import SemanticSegmentationWrapper
from utils.semantic_segmentation_wrapper import SemanticSegmentationWrapper as SSW
import numpy as np
from nadav_color_classiffy import ColorClassifier as ncc
video_file  = '/home/tevel/workspace/data/vid_221014_084538_C.avi'
init_frame  = 1000
cap = cv2.VideoCapture(video_file)
cap.set(1, init_frame-1)
color_class = ColorClassificaton()
old_color_class = ncc(True)
segment_object =  SemanticSegmentationWrapper()
semantic_segmentation_model = SSW()
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
            full_image_color, full_image_fruit_binary, masks_data = semantic_segmentation_model.infer_mask_without_scores(
                frame, [[x,y,r]], True)
            # out, box_out = segment_object.infer(frame,[[x,y,r]] , True)
            color_out = color_class.infer_masked_pixles(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [full_image_fruit_binary])
            crop = frame[y-r:y+r,x-r:x+r,:]
            old_class = old_color_class.grade(crop,masks_data[0]['mask_cropped_according_bbox']==1)
            zeros_matrix = np.zeros_like(frame)
            zeros_matrix[full_image_fruit_binary] = 1
            cv2.imshow('full_mask',zeros_matrix*frame)
            arr8 = (masks_data[0]['mask_cropped_according_bbox']*1).astype(np.uint8)
            mask_colors = cv2.applyColorMap(arr8 * 100, cv2.COLORMAP_HSV)
            # color_out = color_class.infer(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), xywh=[list(roi)])
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