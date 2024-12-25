# Outlook-spam-detection-NN

# Dataset
The dataset used for training is the SMS Spam Collection Dataset, available on Kaggle. This dataset consists of 5,574 SMS messages labeled as either spam or ham (non-spam).

The dataset has the following structure:

Label: "spam" or "ham"
Message: The SMS content
You can download the dataset and place it in the project folder, or use the script to download it automatically.
link: https://www.kaggle.com/datasets/venky73/spam-mails-dataset

# Model
This project uses a simple neural network model built with TensorFlow/Keras. The architecture is as follows:

Input Layer: Vectorized message (converted using TF-IDF or word embeddings)
Hidden Layers: Dense layers with ReLU activation
Output Layer: Sigmoid activation to output the probability of the message being spam

# Model Architecture:
Input: Vectorized text data (TF-IDF or word embeddings)
Layer 1: Dense layer with ReLU activation
Layer 2: Dense layer with ReLU activation
Output: Dense layer with sigmoid activation

# Training
Load the dataset and preprocess the messages (text cleaning, tokenization, and vectorization).
Split the dataset into training and testing sets.
Train the model using the training data.
Evaluate the model on the test data.
The model is trained using binary cross-entropy loss and optimized with Adam optimizer.

# Results
After training:
Accuracy: 0.9903381642512077
Precision: 0.9829351535836177
Recall: 0.9829351535836177
F1-Score: 0.9829351535836177
