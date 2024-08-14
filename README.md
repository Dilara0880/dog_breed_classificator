# Dog Breed Classificator
This repository contains a dog breed classifier that utilizes a pre-trained convolutional neural network, **MobileNetV2**. The model has been fine-tuned on a custom dataset of dog photos, with 70-120 images per class. The current accuracy on the test set is **82.5%**.

### Classification: 
The classifier can currently recognize 120 different dog breeds: affenpinscher, afghan_hound, african_hunting_dog, airedale... 
A full list of recognized breeds can be found in the _breeds.csv_ file.

### Telegram Bot Integration: 
A [Telegram bot What's Dog Breed](https://t.me/whatsdog_bot) is available to provide breed classification. Users can simply send a photo (as an image, not a file) to the bot, and it will return the predicted breed.

###  Performance
Screens of results in Telegram Bot: 

<img width="410" alt="image" src="https://github.com/user-attachments/assets/b6f184f9-ece2-4f1a-9718-bfb6e39d2110">

<img width="387" alt="image" src="https://github.com/user-attachments/assets/1fb12ce3-a42a-4706-a732-1d3c8886c843">

<img width="349" alt="image" src="https://github.com/user-attachments/assets/6ad40afb-81f3-477c-a477-76713f67bb4e">

Results of CNN with **Predicted and Actual breed comparing**
![image](https://github.com/user-attachments/assets/177a317d-3e6a-445f-84bb-5c3a7d2f6c8e)

### Stack:
The project was developed using Python, with TensorFlow and Keras for deep learning and model fine-tuning, OpenCV for image preprocessing, Pandas for data handling, and the Python Telegram Bot API for integrating the classifier into a Telegram bot.
