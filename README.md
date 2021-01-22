# DeepLearningDigitDetection

https://deep-learning-digit-detection.df.r.appspot.com

# Goal

Detect the digit that the user drew on the canva

# Requirements

* Streamlit : to showcase the app
* Streamlit_drawable_canvas : to create the canva on wich the user will draw
* Tensorflow : to build the model and load it inside the app
* OpenCV : to work on the image before feeding it to the model

# How it works

* First, a jupyter notebook is used to create the model
* The model is then loaded into the app
* The user is prompted to draw a digit into the canva
* The drawing is then converted using OpenCV to be feeded to the model
* The result of the prediction is then displayed on screen

# To Do

* Make it more accurate
* Make it prettier
