import telebot, numpy as np, os
from tensorflow import keras
from keras import preprocessing
from PIL import Image
import io
import pandas as pd


token = ''
bot = telebot.TeleBot(token)
welcome_message = '''Hello! 
I can identify 120 dog breeds from your image. 
To start send a picture of a dog'''

wait_for_message = 'Processing'

labels_file = 'training_model/labels.csv'
labels_df = pd.read_csv(labels_file)

labels_df['breed'] = labels_df['breed'].astype('category')
labels_df['breed_code'] = labels_df['breed'].cat.codes


class_indices = {row[1]['breed_code']:row[1]['breed'] for row in labels_df.iterrows()}

model_path = 'dog_breed_classificator/final.keras'

# Define a model
model = keras.models.load_model(model_path)


@bot.message_handler(commands=['start'])
def welcome(message):
    bot.reply_to(message, welcome_message)


@bot.message_handler(content_types=['photo'])
def get_photo(message):
    # Get an image from chat
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)

    # Get a prediction from model
    answer = predict_breed(downloaded_file)

    # Send an answer to the use
    bot.reply_to(message, answer)


@bot.message_handler(func=lambda message: True)
def echo_all(message):
	bot.reply_to(message, 'Send an image to classify dog breed')


def preprocess_image(img, target_size):
    img = img.resize(target_size, Image.NEAREST)
    img_array = preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  
    return img, img_array


def predict_breed(img):
    # Prepare an image
    img = Image.open(io.BytesIO(img))
    img = img.convert('RGB')
    target_size = (128, 128)
    img, img_array = preprocess_image(img, target_size)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_breed = class_indices[predicted_class]
    # Map number to text with dog breed
    return predicted_breed.replace('_', ' ')


bot.polling()
