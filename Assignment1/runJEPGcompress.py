import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt

def jpegCompress(image, quantmatrix):
    '''
        Compress(imagefile, quanmatrix simulates the lossy compression of 
        baseline JPEG, by quantizing the DCT coefficients in 8x8 blocks
    '''
    # Return compressed image in result
    
    H = np.size(image, 0)
    W = np.size(image, 1)

    # Number of 8x8 blocks in the height and width directions
    h8 = int(H / 8)
    w8 = int(W / 8)
    
    # TODO If not an integer number of blocks, pad it with zeros
    result = image.astype('float')
    if h8 != np.ceil(h8) or w8 != np.ceil(w8):
        h8, w8 = int(np.ceil((h8, w8)))
        im = np.zeros((h8 * 8, w8 * 8, 3))
        im[:H, :W, :] = result
        result = im

    # TODO Separate the result into blocks, and compress the blocks via quantization DCT coefficients
    for i in range(0, h8 * 8, 8):
        for j in range(0, w8 * 8, 8):
            mat = np.round(np.divide(cv2.dct(result[i:i+8, j:j+8, 0] - 128), quantmatrix))
            result[i:i+8, j:j+8, 0] = mat
            result[i:i+8, j:j+8, 1] = mat
            result[i:i+8, j:j+8, 2] = mat

    # TODO Convert back from DCT domain to RGB result
    for i in range(0, h8 * 8, 8):
        for j in range(0, w8 * 8, 8):
            mat = cv2.idct(np.multiply(result[i:i+8, j:j+8, 0], quantmatrix)) + 128
            result[i:i+8, j:j+8, 0] = mat
            result[i:i+8, j:j+8, 1] = mat
            result[i:i+8, j:j+8, 2] = mat
    
    return result.astype('uint8')

if __name__ == '__main__':

    im = cv2.imread('./misc/lena_gray.bmp')
    im.astype('float')
    
    quantmatrix = sio.loadmat('./misc/quantmatrix.mat')['quantmatrix']

    out = jpegCompress(im, quantmatrix)

    # Show result, OpenCV is BGR-channel, matplotlib is RGB-channel
    # Or: 
    #   cv2.imshow('output',out)
    #   cv2.waitKey(0)
    plt.imshow(out)
    plt.show()

