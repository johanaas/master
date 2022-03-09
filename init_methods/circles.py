import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import random
from init_methods.utils import decision_function

def not_overlap(point, circles, radius):
    for circle in circles:
        if ((circle[0] - point[0]) ** 2 + (circle[1] - point[1]) ** 2 <= (radius*2) ** 2 ):
            return False
    return True

def append_perturbation(img, mask):
    #cv2.circle(img, (112, 112), 112, (0.5,0.5,0.5), -1 )
    random_noise = np.random.uniform(0, 1, (img.shape))
    #mask = (img == 1)
    adv_sample = copy.deepcopy(img)
    adv_sample[mask] = random_noise[mask]
    return adv_sample


def new_circles(model, params, img, perturbed, main_circle):
    radius = main_circle[3]
    #print("MAIN CIRCLE: ", main_circle[0], main_circle[1])
    #print("Radius: ", main_circle[3])
    #patch_size = 40
    radius_devider = 6

    # Padding to be length of inside-circle radius
    border_padding = ( (radius) / radius_devider )
    #print("border_padding: ", border_padding)

    circles = []
    adv_sample = copy.deepcopy(perturbed)
    # Remove noise from main circle, so little circles can be placed
    adv_sample[main_circle[2]] = img[main_circle[2]]
    
    tries = 0
    
    while True:
        
        # New circle
        length = np.random.uniform(0, (radius))
        angle = np.pi * np.random.uniform(0, 2)

        #print(length)

        x = (np.sqrt(length)* (radius - border_padding)) * np.cos(angle) * 112 + main_circle[0]
        y = (np.sqrt(length)* (radius - border_padding)) * np.sin(angle) * 112 + main_circle[1]

        if not_overlap((x,y), circles, int((radius/radius_devider)*112)) and radius > 0.04:
            cv2.circle(adv_sample, (int(x),int(y)), int((radius/radius_devider)*112), (1,1,1), -1)
            mask = (adv_sample == 1)
            circles.append((x, y, mask, (radius/radius_devider)))
            adv_sample = append_perturbation(adv_sample, mask)
        if tries > 1000: 
            break
        tries += 1
    
    #plt.imshow(adv_sample)
    #plt.show()
    # Now we got many circles inside a circle
    # Converting them to perturbation
    #adv_sample = append_perturbation(img)

    adv_circles = []

    # Looping through circles and removing the not adversarial circles
    for circle in circles:

        #Remove one circle and check if still adversarial
        org_adv_sample = copy.deepcopy(adv_sample)
        adv_sample[circle[2]] = img[circle[2]]
        if decision_function(model, adv_sample[None], params)[0]:
            hei = 0
            #print("Still adversarial => redundant noise: dont need it! ")
        else:
            #print("Not adversarial => important noise: keeping the perturbation.. ")
            adv_sample = org_adv_sample
            adv_circles.append(circle)
    #plt.imshow(adv_sample)
    #plt.show()

    if len(adv_circles) == len(circles):
        print("All circles were important => Cant divide circle, appending whole circle..")
        # We cant divide main_circle into less circles
        # Return the main_circle
        return perturbed, []
    
    #print("adv_circles: ", len(adv_circles))
    return adv_sample, adv_circles
    


def get_circles_perturb(img, model, params, radius=1, plot_each_step=False):
    
    assert img.shape[0] == img.shape[1]
    np.random.seed(69)

    circles = [(112,112, None, radius)]
    #Draw init circle:
    first_img = copy.deepcopy(img)
    cv2.circle(first_img, (112,112), 112, (1,1,1), -1)
    perturbed = append_perturbation(first_img, (first_img == 1))
    assert decision_function(model, first_img, params)

    #plt.imshow(perturbed)
    #plt.show()
    #cv2.circle(img, (112,112), 112, (0,0,0), -1)

    while len(circles) > 0:
        circle = circles.pop(0)
        #append_perturbation(img, circle, circles)
        ##print("Cirlces to investigate: ", len(circles))
        if circle[3] > 0.04:
            perturbed, adv_circles = new_circles(model, params, img, perturbed, circle)

            for adv_circle in adv_circles:
                circles.append(adv_circle)
    return perturbed
