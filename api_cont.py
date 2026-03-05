#usage>streamlit run app.py
import numpy as np

# Function to process the image (can be replaced with your function)
def process_imageEG(image: np.ndarray,preloadedmdl) -> str:
	# Dummy function: returns "bright" or "dark" based on average brightness
	if np.mean(image) > 128:
		return "This looks bright!"
	else:
		return "This looks dark!"

#util
def resize_keep_ratio_height(img, target_h,interpolationway=None):
	h, w = img.shape[:2]
	scale = target_h / h
	new_w = int(w * scale)
	return cv2.resize(img, (new_w, target_h),interpolation=interpolationway)

#mediapipepreprocess part
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------
# 1. Load MediaPipe models (Pose + Hands)
#src>https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#pose_landmarker_model
#src>
# ---------------------------------------------------------

hand_base_options = python.BaseOptions(
	model_asset_path="hand_landmarker.task"
)
hand_options = vision.HandLandmarkerOptions(
	base_options=hand_base_options,
	running_mode=vision.RunningMode.IMAGE
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)


# ---------------------------------------------------------
# 2. Square bounding box helper
# ---------------------------------------------------------
def square_bbox(points, w, h, scale=1.2):
	pts = np.array(points)
	x_min, y_min = pts[:,0].min(), pts[:,1].min()
	x_max, y_max = pts[:,0].max(), pts[:,1].max()

	cx = (x_min + x_max) / 2
	cy = (y_min + y_max) / 2
	side = max(x_max - x_min, y_max - y_min) * scale

	x0 = int(cx - side/2)
	y0 = int(cy - side/2)
	x1 = int(cx + side/2)
	y1 = int(cy + side/2)

	x0 = max(0, x0)
	y0 = max(0, y0)
	x1 = min(w, x1)
	y1 = min(h, y1)

	# enforce square after clamping
	side = min(x1 - x0, y1 - y0)
	return x0, y0, x0 + side, y0 + side


# ---------------------------------------------------------
# 3. Extract keypoints from MediaPipe Tasks
def extract_points(image):
	h, w = image.shape[:2]
	mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
	#mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

	hand_res = hand_detector.detect(mp_image)

	pts = []

	if hand_res.hand_landmarks:
		for hand in hand_res.hand_landmarks:
			for lm in hand:
				pts.append((lm.x * w, lm.y * h))
	return pts


# ---------------------------------------------------------
# 4. Crop to square region around arms + hands
def crop_square(image, target_size=224):
	h, w = image.shape[:2]
	pts = extract_points(image)

	if pts:
		x0, y0, x1, y1 = square_bbox(pts, w, h)
		crop = image[y0:y1, x0:x1]
	else:
		# fallback: center square
		side = min(w, h)
		x0 = (w - side)//2
		y0 = (h - side)//2
		crop = image[y0:y0+side, x0:x0+side]

	crop = cv2.resize(crop, (target_size, target_size))
	return crop


TrgtIMG_SIZE=100
def preprocessimgway5(img):
	img =  crop_square(img, target_size=200)
	#One shot,resize,cvt,normalize
	processedimg=cv2.cvtColor(resize_keep_ratio_height(img , TrgtIMG_SIZE, interpolationway=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype("float32") / 255.0 
	return processedimg

#EVAL EXTRAS
# Function to process the image (can be replaced with your function)
def process_image(image: np.ndarray,preloadedmdl) -> str:
	img_array=preprocessimgway5(image)
	img_array = np.expand_dims(img_array, axis=-1)  # (100, 100, 1)
	img_array = np.expand_dims(img_array, axis=0)  # (1, 100, 100, 1) 
	return predictwcnn(img_array,preloadedmdl)

mdloutmap={0: '0', 1: '1', 2: '10', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9', 11: 'a', 12: 'b', 13: 'c', 14: 'd', 15: 'e', 16: 'f', 17: 'g', 18: 'i', 19: 'k', 20: 'l', 21: 'm', 22: 'n', 23: 'o', 24: 'p', 25: 'q', 26: 'r', 27: 's', 28: 't', 29: 'u', 30: 'v', 31: 'w', 32: 'x', 33: 'z'}

def predictwcnn(processed_image,model):
	# Make a prediction
	predictions = model.predict(processed_image)

	# Get the predicted class (if it's a classification problem)
	predicted_class = np.argmax(predictions, axis=-1).item()  # Get the index of the class with the highest probability then,Convert from numpy array to scalar int

	# If you're using categorical cross-entropy loss, the predictions will be a vector of probabilities(can be used if not use item())
	#print(f"Predicted class index: {predicted_class[0]}")

	# If you want to display the probability of each class (for multi-class classification)
	#print(f"Prediction probabilities: {predictions[0]}")
	return predictions,predicted_class


import time
from PIL import Image
import cv2

import tensorflow as tf


#@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("preprocess5_bslcnn_corehand.keras")# Or 'cnn_model' if using SavedModel format

model = load_my_model()

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_bytes()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

        # run your model
        result_percent, result = process_image(img, model)

        await ws.send_text(str(result))
