from PIL import Image
from SneakerPricePrediction.predict import normalize_pixels
from SneakerPricePrediction.predict import load_df
from StockXScraper import sneaker as scraper
from ./Data-Cleaning/
from pprint import pprint
import pandas as pd
import argparse
import sys
import numpy as np
import autokeras as ak
from tensorflow import keras
import tensorflow as tf
from selenium import webdriver


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser(description='Predict the price of Sneakers')
group = parser.add_mutually_exclusive_group(required=True)

group.add_argument('--webpage', help='URL from www.stockx.com of the sneaker you want to make a prediction of')
group.add_argument('--csv', help='The .csv containing the data you want to visualize')
parser.add_argument('--keras', help='Directory to keras model used to make the prediction',default='./SneakerPricePrediction/autokeras_out/')

args = parser.parse_args(sys.argv[1:])

# load model
model = keras.models.load_model("./SneakerPricePrediction/autokeras_out/", custom_objects=ak.CUSTOM_OBJECTS)
print('\n\n')


print(args)

# check for webpage
if args.webpage != None:
    print(args.webpage)
    driver = webdriver.Firefox()
    shoe_data = scraper.get_shoe_data(args.webpage
            ,driver
            ,'./shoe_images/'
            ,page_wait=1
            ,complex_image_path=False)
    print(shoe_data)
else:
    print('No webpage')
    sys.exit(0)

#store filepath
image_path = shoe_data['image_path']

#crop new sneaker image
crop_image(image_path,directory='./shoe_images/',output_directory='./output_images/')
