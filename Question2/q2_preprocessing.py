#Alex Yeh
#HW4, Question 2

#read in the input image and create the 5D feature vector for each pixel

import cv2
import numpy as np

#read image
img = cv2.imread("208001.jpg", cv2.IMREAD_COLOR)

#print(img.shape)
x_pixels = img.shape[0]
y_pixels = img.shape[1]
nPixels = x_pixels*y_pixels

#5D features
#xloc, yloc, red, green, blue
samples = np.zeros((nPixels,5))

#populate samples
for x in range(x_pixels):
    for y in range(y_pixels):
        p_index = x*y_pixels+y
        rgb = img[x,y]
        features = np.array([x,y,rgb[0],rgb[1],rgb[2]])
        samples[p_index]=features

#linearly normalize each feature entry to [0,1] interval
norm_samples = np.zeros((nPixels,5))
for c in range(samples.shape[1]):
    col = samples[:,c]
    min = np.min(col)
    max = np.max(col)
    norm_samples[:,c] = (col-min)/max

#save preprocessed data
with open('image.npy', 'wb') as f1:
    np.save(f1, norm_samples)

#quick visual check
#print(samples.shape)
#print(samples[300:330])
#print(norm_samples.shape)
#print(norm_samples[300:330])

# Creating GUI window to display an image on screen
cv2.imshow("mushroom", img)
cv2.waitKey(0)
cv2.destroyAllWindows()