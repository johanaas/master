import numpy as np
import matplotlib.pyplot as plt
#from skimage.io import imread, imshow
#from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
#from skimage import color, exposure, transform
#from skimage.exposure import equalize_hist
import cv2
import copy
import query_counter 

def runJPEG(images, ori_img):

    new_images = []

    for img in images:
        transformed_channels = []

        for i in range(3):
            rgb_fft = np.fft.fftshift(np.fft.fft2((img[:, :, i])))
            magnitude = np.log(np.abs(rgb_fft))
            phase = np.angle(rgb_fft, deg=False)

            band_mask = np.ones((224,224))
            cv2.circle(band_mask, (112,112), 100, 0, -1)
            high_mask = (band_mask == 1)
            magnitude[high_mask] = np.zeros(high_mask.shape)[high_mask]

            magnitude = np.exp(magnitude)
            b = magnitude*np.sin(phase)
            a = magnitude*np.cos(phase)
            z = a + b * 1j
            back_shift = np.fft.ifftshift(z)
            bb = np.fft.ifft2(back_shift).real
            transformed_channels.append(bb)
        
        final_image = np.dstack([transformed_channels[0].astype(float), 
                                transformed_channels[1].astype(float), 
                                transformed_channels[2].astype(float)])

        #if query_counter.active_exp == "hsja":
        #    plt.imshow(final_image)
        #    plt.show()

        new_images.append(final_image)

    #plt.imshow(final_image)
    #plt.show()

    return new_images