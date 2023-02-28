import numpy as np


def pre_normalization(im_in):
    sat_indx = np.logical_and(im_in[:,:,0]==255,im_in[:,:,1]==255)
    sat_indx = np.logical_and(sat_indx,im_in[:,:,2]==255)
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
    number_iter =20
    im_in,N_fault = pre_normalization(im_in)
    N = im_in.shape[0]*im_in.shape[1]- N_fault
    for i in range(number_iter):
        im_in = average_pixel(im_in)
        im_in = average_cs(im_in,N)
    # im_in = im_in**2
    return im_in