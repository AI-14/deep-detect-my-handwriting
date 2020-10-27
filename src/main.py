# Author: Amaan Izhar

# Importing the libraries.
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import codecs
import matplotlib.image as mimg
import matplotlib.pyplot as plt
from keras.models import load_model


def display_home_content():
    '''
        Description: Displays the initial content rendered on the home page.
        
        Parameters:
            None
        
        Returns:
            None
    '''

    title = codecs.open('src/markdowns/title.md', 'r', 'utf-8')
    info = codecs.open('src/markdowns/info.md', 'r', 'utf-8')
    design = codecs.open('src/markdowns/design.md', 'r', 'utf-8')
    egimg = Image.open('img/example.png')

    st.markdown(title.read(), unsafe_allow_html=True)
    st.write('\n\n')
    st.markdown(info.read(), unsafe_allow_html=True)
    st.write('\n\n')
    st.image(egimg, width=650, caption='Examples of handwritten digits/alphabets.')
    st.markdown(design.read(), unsafe_allow_html=True)

def predict_digit():
    '''
        Description: Sets up the drawing canvas, saves the image, and predicts the digit based on the choice of model.
        
        Parameters:
            None
        
        Returns:
            None
    '''

    model_choice = st.selectbox('Choose a model to predict', ['Artificial Neural Network', 'Convolutional Neural Network'])
    drawing_mode = st.checkbox("Drawing mode?", True)

    col1, col2 = st.beta_columns(2)

    with col1:
        st.write('Drawing Canvas')

        # Create a canvas component
        image_data = st_canvas(
                15, '#FFFFFF', '#000000', height=280, width=280, drawing_mode=drawing_mode, key="canvas"
        )

    with col2:
        if image_data is not None:
            # Scaling the image.
            image_re = image_data[:, :, :3]
            image_re = image_re.astype(float) // 255
            mimg.imsave('img\\digitImage.png', image_re)
            img = Image.open('img\\digitImage.png').convert('L').resize((28, 28)) 

            st.write('Drawn Image (Scaled)')
            st.image(img, width=280)  # Showing the image drawn.

    show_results = st.button('Show Results')

    # Catching any Exception and passing on it (just in case).
    try:
        if model_choice == 'Artificial Neural Network':
            img_ann = np.array(img).reshape((1, 784)) # Getting the input ready as per ANNs input format.
            model = load_model('src/models/digits_ann.h5')
            pred_ann = model.predict(img_ann)
            if show_results:
                visualize_digits(pred_ann)
    
        if model_choice == 'Convolutional Neural Network':
            img_cnn = np.array(img).reshape((-1, 28, 28, 1)) # Getting the input ready as per CNNs input format.
            model = load_model('src/models/digits_cnn.h5')
            pred_cnn = model.predict(img_cnn)
            if show_results:
                visualize_digits(pred_cnn)
    except Exception:
        pass

def visualize_digits(pred):
    '''
        Description: Generates a bar plot showing the results of the predicted digit.
        
        Parameters:
            pred (numpy.ndarray) - it contains the probabilities of all the predicted digits.
        
        Returns:
            None
    '''

    fig, ax = plt.subplots()
    dig = [x for x in range(10)]
    ax.bar(dig, pred.flatten())
    plt.xticks(dig)
    plt.title('Digit Recognized As')
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    st.pyplot(fig)

def predict_alphabet():
    '''
        Description: Sets up the drawing canvas, saves the image, and predicts the alphabet.
        
        Parameters:
            None
        
        Returns:
            None
    '''

    drawing_mode = st.checkbox("Drawing mode?", True)

    col1, col2 = st.beta_columns(2)

    with col1:
        st.write('Drawing Canvas')

        # Create a canvas component
        image_data = st_canvas(
            15, '#FFFFFF', '#000000', height=280, width=280, drawing_mode=drawing_mode, key="canvas"
        )

    with col2:
        if image_data is not None:

            # Catching any Exception and passing on it (just in case).
            try:
                # Scaling the image.
                image_re = image_data[:, :, :3]
                image_re = image_re.astype(float) // 255  
                mimg.imsave('img/alphaImage.png', image_re)
                img = Image.open('img/alphaImage.png').convert('L').resize((28, 28))

                st.write('Drawn Image (Scaled)')
                st.image(img, width=280)  # Showing the image drawn.
                img = np.array(img).reshape((-1, 28, 28, 1))
                model = load_model('src/models/alphabets_cnn.h5')
                pred = model.predict(img)
            except Exception:
                pass

    show_results = st.button('Show Results')
    if show_results:
        visualize_alphabets(pred)

def visualize_alphabets(pred):
    '''
        Description: Generates a bar plot showing the results of the predicted alphabet.
        
        Parameters:
            pred (numpy.ndarray) - it contains the probabilities of all the predicted alphabets.
        
        Returns:
            None
    '''

    k = [x for x in range(0,26)]
    v = [chr(x) for x in range(65,91)]
    alpha_decoded = dict(zip(k,v))

    fig, ax = plt.subplots()
    ax.bar(alpha_decoded.values(), pred.flatten())
    plt.xticks(k)
    plt.title('Alphabet Recognized As')
    plt.xlabel('Alphabet')
    plt.ylabel('Probability')
    st.pyplot(fig)

def main():
    '''
        Description: All the main functionalities go here. It sets the page_config and calls other 
                     other dependable functions.
        
        Parameters:
            None
        
        Returns:
            None
    '''

    st.beta_set_page_config('Deep Detect My Handwriting', page_icon=':+1:')
    st.sidebar.title('Navigate')
    navigation_mode = st.sidebar.radio('', ['Home 🏠', 'Detect My Digit 🔢', 'Detect My Alphabet 🔤'])

    if navigation_mode == 'Home 🏠':
        display_home_content()
    if navigation_mode == 'Detect My Digit 🔢':
        predict_digit()
    if navigation_mode == 'Detect My Alphabet 🔤':
        predict_alphabet()

# Application starts here.
if __name__ == '__main__':
    main()
