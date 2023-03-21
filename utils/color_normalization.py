import numpy as np
import cv2

def pre_normalization(im_in):
    sat_indx = np.logical_or(im_in[:,:,0]==255,im_in[:,:,1]==255)
    sat_indx = np.logical_or(sat_indx,im_in[:,:,2]==255)
    N = np.sum(sat_indx)
    im_in[sat_indx,:] = 0
    # im_in = im_in.astype(float)
    # im_in = np.sqrt(im_in)
    return im_in, N
def average_pixel(im_in):
    im_tmp = np.sum(im_in,axis=2)+1e-6
    im_out = im_in/np.repeat(np.expand_dims(im_tmp,axis=2),3,axis=2)
    return im_out
def average_cs(im_in,N):
    im_out = np.zeros_like(im_in)
    for ch in range(3):
        mean_val = np.sum(im_in[:,:,ch])
        im_out[:,:,ch] = im_in[:,:,ch]/(mean_val+1e-6)

    return im_out*N/3
def normalization_color_1(im_in):
    number_iter =5
    # im_in,N_fault = pre_normalization(im_in)
    N_fault=0
    N = im_in.shape[0]*im_in.shape[1]- N_fault
    for i in range(number_iter):
        im_in = average_pixel(im_in)
        im_in = average_cs(im_in,N)
    # im_in = im_in**2
    return im_in

def normalization_color_2(img):
    r_norm = cv2.equalizeHist(img[...,0].squeeze())
    g_norm = cv2.equalizeHist(img[...,1].squeeze())
    b_norm = cv2.equalizeHist(img[...,2].squeeze())
    im_out = np.stack([r_norm,g_norm,b_norm],axis=2)
    return im_out

if __name__=='__main__':
    # img_file = '/home/tevel/workspace/data/color_classification/tagged_data/tagged_frames/Stark_LowRes_Level2_C/frame100.jpg'
    img_file ='/home/adi/workspace/RND/data/tagged_frames/Stark_HighRes_Level1_C/frame30.jpg'
    frame = cv2.imread(img_file)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    frame_norm1 = normalization_color_1(frame)
    frame1_uint = frame_norm1*255
    frame1_uint = frame1_uint.astype(np.uint8)
    frame_norm2 = normalization_color_2(frame1_uint)
    cv2.imshow('image_0',frame)
    cv2.imshow('image_1',frame_norm1)
    cv2.imshow('image_2',frame_norm2)
    cv2.waitKey()