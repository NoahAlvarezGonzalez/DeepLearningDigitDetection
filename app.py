import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
from tensorflow.keras.models import load_model
import numpy as np


def main():
    # Load the model
    model = load_model("model/model.h5")

    # HTML used to create the canva
    st.set_page_config(
        page_title="Deep Learning Dgit Detection",
        page_icon=":pencil:",
    )

    hide_streamlit_style = """            
                           <style>            
                           #MainMenu {visibility: hidden;}            
                           footer {visibility: hidden;}            
                           </style>            
                           """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>Deep Learning Digit Detection</h1>", unsafe_allow_html=True)

    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color="#fff",
        background_color="#000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    detect = st.button("Detect")

    if detect:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val = model.predict(test_x.reshape(1, 28, 28))
        st.write(f'result: {np.argmax(val[0])}')


if __name__ == "__main__":
    main()
