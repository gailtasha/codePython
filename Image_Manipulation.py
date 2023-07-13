import os
import numpy as np
from PIL import Image, ImageFilter

os.listdir()

# Basics with numpy and Pillow
image = Image.open('image.jpg')
#image  #to print image
imgarr = np.array(image) #convert image to array
imgarr = imgarr.copy()  #create a copy to manipulate image, don't have permission to modify OG image
print(imgarr.shape)  #last 3 signifies 3 channels
#imgarr  #to print image array
r,g,b = imgarr[:,:,0], imgarr[:,:,1], imgarr[:,:,2] #slicing to different arrays

# imgarr[:,:,1] = np.zeros(g.shape)
# imgarr[:,:,2] = np.zeros(b.shape)  #make g, b channels 0 to show red image
# Image.fromarray(imgarr)  #convert array to image

# imgarr[:,:,0] = np.zeros(r.shape)
# imgarr[:,:,1] = np.zeros(g.shape)  #make r,g channels 0 to show blue image
# Image.fromarray(imgarr)  #convert array to image
# #execute from beginning before manipulating image

# imgarr[:,:,0] = r*2  #multiply r component by 2
# imgarr[:,:,1] = np.zeros(g.shape)
# imgarr[:,:,2] = np.zeros(b.shape)  
# Image.fromarray(imgarr)  #convert array to image

# img = Image.fromarray(imgarr)
# img = img.convert('L')
# img

image = Image.open('image.jpg')
#image  #to print image
imgarr = np.array(image) #convert image to array
imgarr = imgarr.copy()  #create a copy to manipulate image, don't have permission to modify OG image
print(imgarr.shape)  #last 3 signifies 3 channels
#imgarr  #to print image array
r,g,b = imgarr[:,:,0], imgarr[:,:,1], imgarr[:,:,2] #slicing to different arrays
imgarr[:,:,0] = np.minimum(r * 0.393 + g * 0.769 + b * 0.189, 255)                           ##Sepia filter
imgarr[:,:,1] = np.minimum(r * 0.349 + g * 0.686 + b * 0.168, 255)
imgarr[:,:,2] = np.minimum(r * 0.272 + g * 0.534 + b * 0.131, 255)
Image.fromarray(imgarr)



