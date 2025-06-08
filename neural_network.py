import streamlit as st
import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np 

(x_train, y_train), (x_test, y_test)= mnist.load_data()
x_train, x_test= x_train/255.0, x_test/255.0

def train_model():
    model=Sequential([
        Flatten(input_shape=(28,28)),
        Dense(128, activation= 'relu'),
        Dense(10, activation='softmax')

    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3, validation_split=0.1)
    return model
    
@st.cache_resource
def get_model():
    model=train_model()
    return model

model=get_model()

st.title("MNIST Digit Classifier")

index=st.slider("Select test image index", 0, len(x_test)-1, 0)
image=x_test[index]

st.image(image, width=150)
pred=model.predict(image.reshape(1,28,28))
pred_label=np.argmax(pred)

st.write(f"Predicted digit: {pred_label}")
st.bar_chart(pred[0])

