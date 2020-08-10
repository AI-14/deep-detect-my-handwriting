# Importing the libraries.
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import matplotlib.image as mimg
from keras.models import load_model

def init_content():
    # Method for initializing the main page content.

    title = '''
    <div style="background-color:#3f93d9;padding:10px;border-radius:20px">
        <h2 style="color:white;text-align:center;font-size:40px">
            Deep Detect My Handwriting
        </h2>
    </div>
    '''

    info = '''
    A deep learning model for predicting handwritten digits. The model was trained using
    a Deep Feed Forward Neural Network (DFFNN) with an accuracy of 97.9%.

    ***Instructions:***
    - Click on **Drawing Mode** on the sidebar to enter the mode.
    - Draw any digit from 0-9 and click **Predict** button.
    - To redraw, click on **Drawing Mode** to exit. Then double click on the drawn object to 
      remove it. Finally, enter the drawing mode again.
    
    #### Developer: Amaan Izhar [![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat&logo=github)](https://github.com/AI-14)

    > *Note:* Try it 2-3 times for more accurate predictions. In the first few trails, it might not predict
              accurately.
    '''

    design = '''
    <div style="background-color:#6bd6bf;padding:10px;border-radius:20px">
    </div>
    '''
    st.markdown(title, unsafe_allow_html=True)
    st.write('\n\n')
    st.markdown(info)
    st.write('\n\n')
    st.markdown(design, unsafe_allow_html=True)

def prediction():
    # Method for predicting the handwritten digits.

    # Specify the drawing mode.
    drawing_mode = st.sidebar.checkbox("Drawing mode ?", True)

    st.write('Drawing Canvas')

    # Create a canvas component
    image_data = st_canvas(
        15, '#FFFFFF', '#000000', height=280,width=280, drawing_mode=drawing_mode, key="canvas"
    )

    # Predicting the image
    if image_data is not None:
        # Scaling the image.
        image_re = image_data[:,:, :3]
        image_re = image_re.astype(float) // 255
        mimg.imsave('res//images//canvasImage.png', image_re)
        img = Image.open('res//images//canvasImage.png').convert('L').resize((28,28))
        img = np.array(img).reshape((1,784))

        # Loading the model and predicting.
        model = load_model('src//models//handwritten_model.h5')
        pred = model.predict(img)
        if st.button('Predict'):
            digit = pred.argmax()
            st.success(f'You drew the digit {digit}')
        if st.button('Thankyou'):
            st.balloons()

def main():
    # Main method. All the functionalities goes here.
    init_content()
    st.write('\n\n\n\n\n')
    prediction()

# Appllication starts here.
if __name__ == '__main__':
    main()