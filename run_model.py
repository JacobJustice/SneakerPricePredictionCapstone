import sys
import argparse

parser = argparse.ArgumentParser(description='Predict the price of Sneakers')
group = parser.add_mutually_exclusive_group(required=True)

group.add_argument('--webpage', help='URL from www.stockx.com of the sneaker you want to make a prediction of')
group.add_argument('--csv', help='The .csv containing the data you want to visualize')
parser.add_argument('--keras', help='Directory to keras model used to make the prediction',default='./SneakerPricePrediction/autokeras_out/')

args = parser.parse_args(sys.argv[1:])

from PIL import Image
from SneakerPricePrediction.predict import normalize_pixels
from SneakerPricePrediction.predict import load_df
from StockXScraper import sneaker as scraper
from DataCleaning.autocrop import crop_image
from DataCleaning.flatten_images import generate_rgb_row
from pprint import pprint
import pandas as pd
import numpy as np
import autokeras as ak
from tensorflow import keras
import tensorflow as tf
from selenium import webdriver
import joblib

#
# (PS), (GS), (W), (TD), High, Mid, Low
#
def one_hot_encoder(df):
    title = list(df['name'])[0]
    out_dic = {'(PS)':0,
               '(GS)':0,
               '(W)':0,
               '(TD)':0,
               'High':0,
               'Mid':0}

    if '(PS)' in title:
        out_dic.update({'(PS)':1})
    if '(GS)' in title:
        out_dic.update({'(GS)':1})
    if '(W)' in title:
        out_dic.update({'(W)':1})
    if '(TD)' in title:
        out_dic.update({'(TD)':1})
    if 'High' in title:
        out_dic.update({'High':1})
    if 'Mid' in title:
        out_dic.update({'Mid':1})
    if 'Low' in title:
        out_dic.update({'Low':1})
    out_df = pd.DataFrame([out_dic])
    out_df = df.merge(out_df,how='left',left_index=True,right_index=True)

    return out_df

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# load model
model = keras.models.load_model("./SneakerPricePrediction/autokeras_out/", custom_objects=ak.CUSTOM_OBJECTS)
print('\n\n')


shoe_data = {}
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
    driver.close()
else:
    print('No webpage')
    sys.exit(0)

#store filepath
image_path = shoe_data['image_path']
image_fn = image_path[image_path.rfind('/')+1:]
image_dir = image_path[:image_path.rfind('/')+1]
output_directory = './output_images/'

print()
print("Cropping downloaded image...")
#crop new sneaker image
crop_image(image_fn,directory=image_dir,output_directory=output_directory)

print("Flattening cropped image...")
#generate_dict with rgb info
rgb_row_dict = generate_rgb_row(image_fn[:-4]+'.png', directory_path=output_directory)
#pprint(rgb_row_dict)
print("flattened image!")

print("Making DataFrame...")

shoe_df = pd.DataFrame([shoe_data])
rgb_df = pd.DataFrame([rgb_row_dict])

df = shoe_df.merge(rgb_df,how='left',left_index=True,right_index=True)

# encode name
df = one_hot_encoder(df)
#print(df)

# convert retail_price to dollars
df['retail_price'] = df['retail_price'].str[1:]
df['retail_price'] = df['retail_price'].str.replace(',','')
df['retail_price'] = df['retail_price'].astype(int)

# drop features not used for predictions
#df = df.drop(['url', 'image_path','colorway','release_date','filename','style_code','name','price_premium'], axis=1)

df = normalize_pixels(df)
df = df.drop(['average_sale_price', 'ticker'], axis=1)
#print(df)

print("Your predicted average_sale_price is: ", model.predict(df)[0][0])
