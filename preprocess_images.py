import numpy as np 
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import os
import scipy 
from scipy import ndimage
import tqdm
from tqdm import tqdm

def preprocess_images(image_dir,save_dir,image_size,image_channels=3,extension=".png"):
    print("processing....")
    training_data = []
    if(not os.path.exists(save_dir)):
        os.mkdir(save_dir)
    if(os.path.exists(image_dir)):
        print(len(os.listdir(image_dir))," images to resized and processed")
        print("\n Processing images....")
        for index ,filename in enumerate(tqdm(os.listdir(image_dir))):
        
            path = os.path.join(image_dir,filename)
            image = Image.open(path).resize((image_size,image_size),Image.ANTIALIAS)
            image_arr  = np.asarray(image)
            if(image_arr.shape == (image_size,image_size,image_channels)):
                training_data.append(image_arr)
                im = Image.fromarray(image_arr)
                im.save(save_dir+"/image_"+str(index)+extension)
        print(training_data[0].shape)
        training_data = np.reshape(training_data,(-1,image_size,image_size,image_channels))
        training_data = training_data.astype(np.float32)
        print("Successfully processed and reshaped images")
    else:
        print("Image data directory does not exist")
        print("\nPlease check if specified training data directory path is correct")
    return None


try :
    image_dir = sys.argv[1] #Relative path of image directory
    save_dir  = sys.argv[2] # Relative path of save directory
    image_shape = int(sys.argv[3]) #image shape an integer


    print("Image source directory : ",os.path.join(os.getcwd(),image_dir))
    print("Saving processed images in  : ",os.path.join(os.getcwd()),save_dir)
    print(f"Resized image shape : ({image_shape},{image_shape},{3})")
except IndexError as error:
    print("Error")
    if(len(sys.argv) == 0):
        print("No parameters are specified")
    elif(len(sys.argv) == 1):
        print("Save directory , image shape and image_channel parameters are not specified")
    elif(len(sys.argv) == 2):
        print("image shape and image channel parameters not specified")
    elif(len(sys.argv) == 3):
        print("image channel parameter not specified")

preprocess_images(image_dir=image_dir,save_dir=save_dir,image_size=image_shape)
