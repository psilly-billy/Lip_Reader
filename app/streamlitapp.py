import streamlit as st
import os
import imageio
import ffmpeg

import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the Streamlit app as wide
st.set_page_config(layout='wide')

# setup Sidebar
with st.sidebar:
    
    st.image('1975577816_mirror5.jpg')
             
    st.title("Lip Reader")
    st.info(' Aplication originally developed from the LipNet deep learning model.')
    

#Generate a list of options or videos
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns
col1, col2 = st.columns(2)

if options:
    
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        #make path for selected video
        file_path = os.path.join('..','data','s1', selected_video)
        #convert video to mp4
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)
        
    with col2:
        st.info("What the ML nodel sees when making a prediction")
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('gifanimation.gif', video, fps=15)
        st.image('gifanimation.gif', width=400)
        
        
        
        st.info("Output from the ML model as Token ")
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)
        
        
        
        st.info('Decode token into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        
        
        
    