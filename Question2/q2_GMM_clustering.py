#Alex Yeh
#HW4 Q2
#uses GMM clustering to segment an image

import numpy as np
import cv2
from sklearn.mixture import GaussianMixture

#model order selected
order=5
image = 'image'

#train using all samples
samples = np.load(open(image+'.npy', 'rb'))
N = samples.shape[0]
model = GaussianMixture(n_components=order)
model.fit(samples)

#predict labels for all samples
labels = model.predict(samples)
#print(np.unique(labels))
#translate labels into grayscale (between 0 and 1)
grayscale = (labels - np.min(labels))/np.max(labels)
#print(np.unique(grayscale))


#create segmented image 
x_pixels = len(np.unique(samples[:,0]))
y_pixels = len(np.unique(samples[:,1]))

segmented_image = np.zeros((x_pixels,y_pixels,1))
for x in range(x_pixels):
    for y in range(y_pixels):
        p_index = x*y_pixels+y
        segmented_image[x,y] = grayscale[p_index]



#show original image
img = cv2.imread("208001.jpg", cv2.IMREAD_COLOR)
cv2.imshow("Original Mushroom", img)

#show segmented image
cv2.imshow("Segmented Mushroom", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()