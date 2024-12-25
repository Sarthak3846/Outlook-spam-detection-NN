import numpy as np
import pandas as pd 
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("C:\\Users\\Sarthak Tyagi\\Downloads\\spam_ham_dataset.csv")
df = df.drop(['Unnamed: 0','label'],axis='columns')

def split_subject_body(text):
    if "Subject:" in text:
        parts = text.split("Subject:", 1)[1].split("\n", 1)
        subject = parts[0].strip()
        body = parts[1].strip() if len(parts) > 1 else ""
    else:
        subject = ""
        body = text.strip()
    return pd.Series([subject, body])
df[["subject", "body"]] = df["text"].apply(split_subject_body)
df = df.drop(columns=["text"])

from sklearn.feature_extraction.text import TfidfVectorizer 

tfidf_subject = TfidfVectorizer(max_features=1000)
tfidf_body = TfidfVectorizer(max_features=4000) 
subject_tfidf = tfidf_subject.fit_transform(df['subject']).toarray()
body_tfidf = tfidf_body.fit_transform(df['body']).toarray()

X = np.hstack((subject_tfidf, body_tfidf))
y = df['label_num'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = Sequential([
    Dense(64,input_dim=X_train.shape[1],activation='relu'),
    Dropout(0.5),
    Dense(32,activation='relu'),
    Dropout(0.5),
    Dense(1,activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train,y_train,epochs=10,batch_size=32,validation_split=0.2)

y_pred = (model.predict(X_test) > 0.5).astype(int)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1-Score: {f1_score(y_test, y_pred)}")

