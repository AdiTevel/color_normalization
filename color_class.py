import tensorflow as tf
from keras.models import Model,load_model
from utils.semantic_segmentation_utils import SemanticSegmentationWrapper
import numpy as np
import onnxruntime as rt
import time
from utils.color_normalization import normalization_color_1 as normalization
MODEL_PATH = 'models/2layers_dense_64_h.h5'
ONNX_PATH = 'models/color_classifier.onnx'

predefined_bins = np.arange(2, 254, 2)



class ColorClassificaton:
    def __init__(self,mode= 'tf'):
        self.mode = mode
        self.load_model()
        self.segment_object =  SemanticSegmentationWrapper()

    def load_model(self):
        if self.mode == 'tf':
            self.model =load_model(MODEL_PATH)
            self.classify = self.classify_tf


        elif self.mode == 'onnx':
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cuda_mem_limit': 2 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider',
            ]

            self.model = rt.InferenceSession(ONNX_PATH,providers= providers)#['CUDAExecutionProvider','CPUExecutionProvider'])
            # self.model = rt.InferenceSession(ONNX_PATH,providers= ['CUDAExecutionProvider','CPUExecutionProvider'])
            self.classify = self.classify_onnx
    def histogram_generator(self,normalized_crop):

        hist_r, bin_edges = np.histogram(normalized_crop[:,0],
                                         bins=predefined_bins
                                         )
        hist_g, bin_edges = np.histogram(normalized_crop[:,1],
                                         bins=predefined_bins
                                         )
        hist_b, bin_edges = np.histogram(normalized_crop[:,2],
                                         bins=predefined_bins
                                         )
        hist_r = hist_r / np.sum(hist_r)
        hist_g = hist_g / np.sum(hist_g)
        hist_b = hist_b / np.sum(hist_b)
        return np.concatenate((hist_r,hist_g,hist_b))

    def infer_masked_pixles(self,image,mask_list):
        image_normalized = normalization(image)

        img_norm_uint = 255 * image_normalized
        img_norm_uint = img_norm_uint.astype(np.uint8)

        X = []
        for mask in mask_list:

            normalized_crop = img_norm_uint[mask==1, :]
            X.append(self.histogram_generator(normalized_crop))
        X = np.array(X)
        classification = self.classify(X)
        return classification

    def infer_image_bbox(self,image,bbox_list):
        image_normalized = normalization(image)

        img_norm_uint = 255 * image_normalized
        img_norm_uint = img_norm_uint.astype(np.uint8)

        X =[]
        for bbox in bbox_list:
            normalized_crop = img_norm_uint[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2],:]
            normalized_crop = normalized_crop.reshape((bbox[3]*bbox[2],3))
            X.append(self.histogram_generator(normalized_crop))
        X = np.array(X)
        classification = self.classify(X)
        return classification

    def classify_tf(self,hist):
        class_predicted = self.model.predict(hist)
        return class_predicted

    def classify_onnx(self,hist):
        class_predicted = self.model.run(None, {"input_5": hist.astype(np.float32)})[0]
        return class_predicted

    def infer(self,img,xyr=None, xywh = None):
        if xyr is not None:
            detection = xyr
        elif xywh is not None:
            detection = []
            for bbox in xywh :
                x = bbox[0] + int(bbox[2] / 2)
                y = bbox[1] + int(bbox[3] / 2)
                r = int(max(bbox[2] / 2, bbox[3] / 2))
                detection.append([x,y,r])

        else:
            assert('No valid input detection in image')

        out, box_out = self.segment_object.infer(img, detection, True)
        classes_segmented = self.infer_masked_pixles(img, box_out)
        return classes_segmented

if __name__=='__main__':
    import cv2
    import pandas as pd
    from os import path
    # from skimage import io, color
    import matplotlib.pyplot as plt
    from utils.semantic_segmentation_utils import SemanticSegmentationWrapper
    # fn = '/home/adi/workspace/RND/data/tagged_frames/Stark_HighRes_Level1_2_C/frame20.jpg'
    fn = '/home/tevel/workspace/data/color_classification/tagged_data/tagged_frames/Stark_MedRes_Level2_2_C/frame100.jpg'
    # dir = '/home/adi/workspace/RND/data/tagged_frames/Stark_MedRes_Level2_2_C'
    # dir = '/home/adi/workspace/RND/data/tagged_frames/Stark_LowRes_Level4_2_C'
    # dir = '/home/adi/workspace/RND/data/tagged_frames/Stark_MedRes_Level3_2_C'
    dir = path.dirname(fn)
    img = cv2.imread(fn)
    y_origin,x_origin,_ = img.shape
    img = cv2.resize(img,(640,480), interpolation = cv2.INTER_AREA)
    x_scale = 640./x_origin
    y_scale = 480./y_origin
    img_bgr = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img_for_imshow = np.copy(img)
    ann = pd.read_csv(path.join(dir, 'ann.csv'))
    # Blue color in BGR
    clr = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    bbox_list =[]
    for i in range(len(ann)):
        fn_ = ann.loc[i, 'frame']
        bbox = ann.loc[i, '[\'x, y, w, h\']']
        bbox = bbox[1:-1].split(', ')
        bbox = [int(i) for i in bbox]
        bbox = [int(bbox[0]*x_scale),int(bbox[1]*y_scale) ,int(bbox[2]*x_scale),int(bbox[3]*y_scale)]
        img_for_imshow = cv2.rectangle(img_for_imshow,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),clr,thickness)
        bbox_list.append(bbox)
    # seg_obj = SemanticSegmentationWrapper()
    # bbox = bbox_list[6]
    # x = bbox[0] + int(bbox[2]/2)
    # y = bbox[1] + int(bbox[3]/2)
    # r = int(max(bbox[2]/2,bbox[3]/2))
    # out,box_out = seg_obj.infer(img,[[x,y,r]],True )
    # # cv2.imshow('figure',img_for_imshow)
    # # cv2.waitKey()
    c_classifier_tf = ColorClassificaton('tf')
    c_classifier_onnx = ColorClassificaton('onnx')

    # # io.imshow(color.label2rgb(out[0], img_for_imshow, colors=[(255, 0, 0), (0, 0, 255)], alpha=0.01, bg_label=0, bg_color=None))
    # plt.imshow(img ,cmap='gray')
    # plt.imshow(box_out[0], cmap='jet', alpha=0.5)  # interpolation='none'
    # plt.show()
    # classes_bbox = c_classifier.infer_image_bbox(img,bbox_list)
    # classes_segmented = c_classifier.infer_masked_pixles(img,box_out)
    classes_segmented_tf =  c_classifier_tf.infer(img,xywh=bbox_list)
    classes_segmented_onnx =  c_classifier_onnx.infer(img,xywh=bbox_list)
    t1 = time.time()
    print(classes_segmented_tf,classes_segmented_onnx)
    t2 = time.time()
    print(classes_segmented_tf-classes_segmented_onnx)
    t3 = time.time()
    print(f'tf _time  = {t2-t1} \n onnx_time = {t3-t2}')



