import numpy as np

def k_delta(x):
    if abs(x)<0.1:
        return 1
    else:
        return 0

def k(r): # try also the Epanechnikov kernel [5] D. Comaniciu, V. Ramesh and P. Meer, Real-Time Tracking of Non-Rigid Objects using Mean Shift, Computer Vision and Pattern Recognition (2000) 142-149.
    if r<1:
        return 1-r**2
    else:
        return 0

def B_coefficient(a,b):
    return np.sum(np.sqrt(a*b))

def get_weight(B):
    #sigma=0.1
    sigma=0.02
    return np.exp(-(1.-B)/sigma)
    #return 1./(np.sqrt(2*np.pi)*sigma)*np.exp(-(1.-B)/(2*sigma**2))

def get_pixels(img,x,y,h_x,h_y):
    crop_img=img[y-h_y:y+h_y,x-h_x,x+h_x]
    return crop_img

def get_random_index(weights):
    weights_acc=[0]
    index=[]
    for i in range(len(weights)):
        weights_acc.append(weights_acc[i]+weights[i])
    for e in range(len(weights)):
        r=np.random.rand()
        for i in range(len(weights)):
            if r>weights_acc[i] and  r<weights_acc[i+1]:
                index.append(i)
                continue
    return index
