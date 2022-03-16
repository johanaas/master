import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist
from utils import decision_function
import cv2
import scipy.fftpack as fp
import copy




def get_fourier_perturbation(img, model, params):

    bgr_img = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2BGR)

    img_yuv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)

    # Lets mess around with y

    f = np.fft.fft2(y)
    f_shift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(f_shift))

    #plt.imshow(y, cmap='gray')
    #plt.imshow(magnitude_spectrum, cmap='gray')
    #plt.show()
    radius = 26
    while radius > 2:

        band_mask = np.zeros((224,224))
        #print(f_shift.shape, band_mask.shape)
        band_mask[110:114,110:114] = 1
        cv2.circle(band_mask, (112,112), radius, 1, -1)
        #plt.imshow(band_mask)
        #plt.show()

        zero_mask = (band_mask != 1)

        f_perturb = f_shift
        f_perturb[zero_mask] = 0
        #f_mag = np.log(np.abs(f_perturb))
        #plt.imshow(f_mag, cmap='gray')
        #plt.show()

        back_shift = np.fft.ifftshift(f_perturb)
        back_y = np.float32(np.fft.ifft2(back_shift).real) # abs and float32
        

        #print(back_y)
        img_merge = np.dstack([back_y, u, v])  #cv2.merge((back_y,u,v)) #np.dstack([back_y, u, v]) 
        #print("img_merge- max, min: ", np.max(img_merge), np.min(img_merge))
        bgr_img_back = cv2.cvtColor(np.float32(img_merge), cv2.COLOR_YUV2BGR)
        #print("max, min: ", np.max(bgr_img_back), np.min(bgr_img_back))
        img_back = cv2.cvtColor(np.float32(bgr_img_back), cv2.COLOR_BGR2RGB)
        img_back = np.clip(img_back, 0,1)

        #print("max, min: ", np.max(img_back), np.min(img_back))
        #plt.imshow(img_back)
        #plt.show()

        if decision_function(model,img_back[None], params)[0]:
            # Adversarial
            return img_back
        else:
            radius -= 2
            print("Radius: ", radius)
            #plt.imshow(img_back)
            #plt.show()
            if radius == 2:
                f_perturb = np.zeros((224,224))
                back_shift = np.fft.ifftshift(f_perturb)
                back_y = np.float32(np.fft.ifft2(back_shift).real)
                img_merge = np.dstack([back_y, u, v])
                bgr_img_back = cv2.cvtColor(np.float32(img_merge), cv2.COLOR_YUV2BGR)
                img_back = cv2.cvtColor(np.float32(bgr_img_back), cv2.COLOR_BGR2RGB)
                img_back = np.clip(img_back, 0,1)

    return img_back


def make_lut_u():
    return np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)

def make_lut_v():
    return np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)


def fourier_transform_rgb(img, model, params):
    
    ## Functions to go from image to frequency-image and back
    im2freq = lambda data: fp.rfft(fp.rfft(data, axis=0),
                                axis=1)
    freq2im = lambda f: fp.irfft(fp.irfft(f, axis=1),
                                axis=0)

    ## Read in data file and transform
    data = img #.astype(float) / 255 #np.array(Image.open('test.png'))

    freq = im2freq(data)
    back = freq2im(freq)
    # Make sure the forward and backward transforms work!
    assert(np.allclose(data, back))

    

    
    ## Helper functions to rescale a frequency-image to [0, 255] and save
    remmax = lambda x: x/x.max()
    remmin = lambda x: x - np.amin(x, axis=(0,1), keepdims=True)
    touint8 = lambda x: (remmax(remmin(x))*(256-1e-4)).astype(int)

    #def arr2im(data, fname):
    #    out = Image.new('RGB', data.shape[1::-1])
    #    out.putdata(map(tuple, data.reshape(-1, 3)))
    #    out.save(fname)

    #arr2im(touint8(freq), 'freq.png')
    #print("max, min: ", np.max(freq), np.min(freq))
    scaled_freq = touint8(freq)
    #plt.imshow(scaled_freq)
    #plt.show()
    shifted = np.fft.fftshift(scaled_freq)
    #plt.imshow(shifted)
    #plt.show()

    """

    f = np.fft.fft2(cv2.cvtColor(np.float32(data), cv2.COLOR_RGB2GRAY))
    f_shift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(f_shift))
    print("max, min: ", np.max(magnitude_spectrum), np.min(magnitude_spectrum))
    plt.imshow(magnitude_spectrum)
    plt.show()
    mag_back = np.fft.ifftshift(magnitude_spectrum)
    #img_back = np.fft.ifft2(mag_back, cv2.COLOR_GRAY2RGB)
    img_back = np.fft.ifft2(f, cv2.COLOR_GRAY2RGB)
    plt.imshow(img_back)
    plt.show()

    """
    
    


    #img = cv2.imread('shed.png')
    bgr_img = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2BGR)

    img_yuv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)

    # Lets mess around with y

    f = np.fft.fft2(y)
    f_shift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(f_shift))

    #plt.imshow(y, cmap='gray')
    #plt.imshow(magnitude_spectrum, cmap='gray')
    #plt.show()

    band_mask = np.zeros((224,224))
    #print(f_shift.shape, band_mask.shape)
    band_mask[110:114,110:114] = 1
    cv2.circle(band_mask, (112,112), 25, 1, -1)
    #plt.imshow(band_mask)
    #plt.show()

    zero_mask = (band_mask != 1)

    f_perturb = f_shift
    f_perturb[zero_mask] = 0
    #f_mag = np.log(np.abs(f_perturb))
    #plt.imshow(f_mag, cmap='gray')
    #plt.show()

    back_shift = np.fft.ifftshift(f_perturb)
    back_y = np.float32(np.abs(np.fft.ifft2(back_shift)))

    #print(back_y.shape)
    #print("max, min: ", np.max(back_y), np.min(back_y))
    #print(y.shape)
    #print("max, min: ", np.max(y), np.min(y))
    #print(u.shape)
    #print("max, min: ", np.max(u), np.min(u))
    #print(v.shape)
    #print("max, min: ", np.max(v), np.min(v))

    img_merge = np.dstack([back_y, u, v])  #cv2.merge((back_y,u,v)) #np.dstack([back_y, u, v]) 
    img_back = np.abs(cv2.cvtColor(img_merge, cv2.COLOR_YUV2RGB))

    #print("max, min: ", np.max(img_back), np.min(img_back))
    #plt.imshow(img_back)
    #plt.show()

    return img_back

    #print("max, min: ", np.max(scaled_freq), np.min(scaled_freq))
    #print("max, min: ", np.max(scaled_freq), np.min(scaled_freq))
    #log_scaled_freq = np.log(scaled_freq)
    #plt.imshow(log_scaled_freq)
    #plt.show()
    #plt.imshow(scaled_freq)
    #plt.show()
    

    #band_pass_mask = np.ones((224,224,3))
    
    # First filter: 8.2
    #band_pass_mask[28:56,28:56,:] = 0
    
    # Second filter: 
    #band_pass_mask[14:28,:28,:] = 0
    #band_pass_mask[:28,14:28,:] = 0

    # Finding optimal band pass
    """
    widthx = 16

    startx = 16


    
    for j in range(1,10):
        band_pass_mask = np.ones((224,224,3))

        startx += j+8
        print("startx: ", startx)
        for i in range(0,startx,int(widthx/2)):
            band_pass_mask[ i:i+widthx, (startx - widthx)-i:startx-i,:] = 0
            img_back = touint8(freq2im(freq * band_pass_mask))
        #print("Adversarial: ", decision_function(model,img_back[None] / 255, params)[0])
        #plt.imshow(band_pass_mask)
        #plt.show()
        # Check if perturbation is not adversarial
        if not decision_function(model,img_back[None] / 255, params)[0]:
            # We have gone one step too far
            band_pass_mask = np.ones((224,224,3))
            
            startx = startx - (j-1)-8
            print("startx: ", startx)
            for i in range(0,startx,int(widthx/2)):
                band_pass_mask[ i:i+widthx, (startx - widthx)-i:startx-i,:] = 0
                img_back = touint8(freq2im(freq * band_pass_mask))
            #plt.imshow(band_pass_mask)
            #plt.show()
            print("Adversarial: ", decision_function(model,img_back[None] / 255, params)[0])
            break
    """
    # Convert img to luminace


    # Find maximum noise
    band_pass_mask = np.zeros((224,224,3))
    band_pass_mask[:2,:2,:] = 1
    band_pass_mask[200:,200:,:] = 1
    img_back = touint8(freq2im(freq * band_pass_mask))
    plt.imshow(img_back)
    plt.show()
    return img_back

    mask_size = 16
    startx = mask_size
    while mask_size < 224:
        band_pass_mask = np.ones((224,224,3))
        startx += mask_size
        print("startx: ", startx)
        #Make band filter
        for i in range(0,startx,int(mask_size/2)):
            band_pass_mask[ i:i+mask_size, (startx - mask_size)-i:startx-i,:] = 0
            img_back = touint8(freq2im(freq * band_pass_mask))
    
        # Check if perturbation is not adversarial
        if not decision_function(model,img_back[None] / 255, params)[0]:
            if startx == (mask_size+mask_size):
                # If first step: increase mask_size
                mask_size = mask_size * 2
                startx = mask_size
                print("startx: ", startx)
            else:
                # Take one step back: Cause we know thats adversarial
                startx -= mask_size
                band_pass_mask = np.ones((224,224,3))
                print("startx: ", startx)
                for i in range(0,startx,int(mask_size/2)):
                    band_pass_mask[ i:i+mask_size, (startx - mask_size)-i:startx-i,:] = 0
                    img_back = touint8(freq2im(freq * band_pass_mask))
                print("Adversarial: ", decision_function(model,img_back[None] / 255, params)[0])
                break

    #plt.imshow(band_pass_mask)
    #plt.show()

    #img_back = touint8(freq2im(freq * band_pass_mask))
    #print("max, min: ", np.max(img_back), np.min(img_back))
    #img_back = output.shape[1::-1]
    #print("Adversarial: ", decision_function(model,img_back[None] / 255, params)[0])
    #plt.imshow(img_back)
    #plt.show()

    """
    # Make frequency-image of cat photo
    freq = im2freq(np.array(Image.open('cat.jpg')))

    # Load three frequency-domain masks (DSP "filters")
    bpfMask = np.array(Image.open('cat-mask-bpfcorner.png')).astype(float) / 255
    hpfMask = np.array(Image.open('cat-mask-hpfcorner.png')).astype(float) / 255
    lpfMask = np.array(Image.open('cat-mask-corner.png')).astype(float) / 255

    # Apply each filter and save the output
    arr2im(touint8(freq2im(freq * bpfMask)), 'cat-bpf.png')
    arr2im(touint8(freq2im(freq * hpfMask)), 'cat-hpf.png')
    arr2im(touint8(freq2im(freq * lpfMask)), 'cat-lpf.png')
    """










    """
    
    
    
    #img = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)
    #dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    #dft_shift = np.fft.fftshift(dft)

    rows, cols, channels = img.shape
    crow,ccol = int(rows/2) , int(cols/2)

    transformed_channels = []

    for i in range(3):
        rgb_fft = np.fft.fftshift(np.fft.fft2((img[:, :, i])))
        mask = np.zeros((rows,cols),np.uint8)
        mask[crow-30:crow+30, ccol-30:ccol+30] = 1

        #rgb_fft = rgb_fft * mask
        fshift = rgb_fft*mask
        f_ishift = np.fft.ifftshift(fshift)
        #img_back = cv2.idft(f_ishift)

        transformed_channels.append(cv2.idft(f_ishift))

    img_back = np.dstack([transformed_channels[0].astype(int), 
                             transformed_channels[1].astype(int), 
                             transformed_channels[2].astype(int)])
    
    print("Adversarial: ", decision_function(model,img_back[None], params)[0])
    plt.imshow(img_back)
    plt.show()
    """
    """                        
    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1

    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

    #print("Adversarial: ", decision_function(model,img_back[None], params)[0])

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    """

    """
    f_size = 25
    transformed_channels = []

    # View image and fourier as grey_img
    dark_image_grey = rgb2gray(image)
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(dark_image_grey))
    plt.imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
    plt.show()

    for i in range(3):
        rgb_fft = np.fft.fftshift(np.fft.fft2((image[:, :, i])))
        print(rgb_fft)
        #rgb_fft[:225, 235:237] = 1
        #rgb_fft[-225:,235:237] = 1
        rgb_fft[102:122, 102:122] = 0.5
        rgb_fft[102:122, 102:122] = 0.5
        print(rgb_fft.shape)
        #mask = (rgb_fft < 0.8)
        #mask1 = (rgb_fft > 0.2)
        #rgb_fft[mask] = 1
        #rgb_fft[mask1] = 1
        transformed_channels.append(abs(np.fft.ifft2(rgb_fft)))
    
    final_image = np.dstack([transformed_channels[0].astype(int), 
                             transformed_channels[1].astype(int), 
                             transformed_channels[2].astype(int)])
    
    fig, ax = plt.subplots(1, 2, figsize=(17,12))
    ax[0].imshow(image)
    ax[0].set_title('Original Image', fontsize = f_size)
    ax[0].set_axis_off()
    
    ax[1].imshow(final_image)
    ax[1].set_title('Transformed Image', fontsize = f_size)
    ax[1].set_axis_off()
    
    fig.tight_layout()
    print("Adversarial: ", decision_function(model,final_image[None], params)[0])
    plt.show()
    """
    return img_back / 255