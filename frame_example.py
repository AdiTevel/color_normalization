import cv2
from color_class import ColorClassificaton
img_file  = '/home/tevel/Downloads/Screenshot 2023-02-15 at 04-51-37 NORMA GALA XFC 21-22 v2.pdf.png'
frame = cv2.imread(img_file)
frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

color_class = ColorClassificaton()

roi = cv2.selectROI(frame)
color_out = color_class.infer_image_bbox(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR),[list(roi)])
# color_out = color_class.infer_image_bbox(frame,[list(roi)])
cv2.rectangle(frame,(roi[0],roi[1]),(roi[0]+roi[2],roi[1]+roi[3]),(255,0,0),2)
cv2.putText(frame, str(color_out), (roi[0], roi[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
cv2.imshow('Frame', frame)
cv2.waitKey()