# Lip Reader

This repository contains code for a Lip Reader application. The application uses a deep learning model to predict spoken words from lip movements in a video.

## Files

- `modelutil.py`: Contains utility functions for loading the deep learning model.
- `utils.py`: Contains utility functions for data processing.
- `record_test.py`: Test script for recording lip movements and saving them to a video file.
- `streamlitapp.py`: Streamlit application script for visualizing the lip movements and running the prediction model.
- `test_video.mp4`: Sample video file for testing the lip reading model.
- `data/`: Directory containing the training data used for training the lip reading model.
- `model/`: Directory containing the trained model weights.
- `Lip_reading.ipynb`: Jupyter notebook documenting the training process.

## Dependencies

The following dependencies are required to run the code:

- Python 3.x
- TensorFlow
- OpenCV
- dlib
- json
- imageio
- ffmpeg
- streamlit


## Usage

### Recording Lip Movements

To record lip movements and save them to a video file, run the `record_test.py` script. Make sure you have a webcam connected to your computer.

```bash
python record_test.py
```
The recorded video will be saved as recorded_video/output.mp4.

Running the Lip Reader Application
To run the Lip Reader application, execute the streamlitapp.py script.

Create File
```bash
streamlit run streamlitapp.py
```
The application will open in your web browser. Select a video from the dropdown menu and click on the "Run Prediction" button to visualize the lip movements and see the predicted words.

Training
If you're interested in the training process, you can refer to the `Lip_reading.ipynb` notebook. It provides a detailed explanation of the steps involved in training the lip reading model.

Model
The lip reading model is implemented in the `utils.py` and `modelutil.py` files. The load_model function in `modelutil.py` loads the pre-trained model weights from the `model/ directory` and returns the model instance.

License
This code is released under the MIT License.

Feel free to modify and use this code for your projects. If you find it helpful, a star to this repository would be greatly appreciated!





