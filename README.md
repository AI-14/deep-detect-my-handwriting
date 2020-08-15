# Deep Detect My Handwriting
  ![Python](https://img.shields.io/badge/-Python-black?style=flat&logo=python)
  ![Deep Learning](https://img.shields.io/badge/-Deep%20Learning-566be8?style=flat)
  ![Sklearn](https://img.shields.io/badge/-Sklearn-1fb30e?style=flat)
  ![Tensorflow](https://img.shields.io/badge/-Tensorflow-gray?style=flat&logo=tensorflow)
  ![Keras](https://img.shields.io/badge/-Keras-gray?style=flat&logo=keras)
  ![Streamlit](https://img.shields.io/badge/-Streamlit-f0806c?style=flat)

## Description
   A streamlit app for predicting handwritten drawn images using deep learning model (DFFNN). All you have to do is draw any digit from 0-9 on the canvas and the
   model will predict the digit you drew in real time.
   
## Screenshots Of The Application
![](/res/readme_res/Pic1.png)

![](/res/readme_res/Pic2.png)

![](/res/readme_res/Pic3.png)

## Installation And Usage
1. Installation
   - Download/clone this repository and create a proper project folder where you will extract this repo's contents. Then open terminal (make sure you are in the project's directory).
   - Create a virtual environment using the command ````py -m venv yourVenvName```` and activate it using ````yourVenvName\Scripts\activate.bat````.
   - Then run the following command ````pip install -r requirements.txt````. With this, all the dependencies will be installed in your virtual environment. 
> **Note:** *If any dependency is missing or an error shows up, install it using ````pip install moduleName````*.

2. Usage
   - Open your project folder and go to the terminal and activate your virtual environment. Then type ````streamlit run src\main.py```` and the app will open in your web 
   browser. Now you can interact with it or play with the code and add your own features and if you wish - you can deploy it on heroku.
