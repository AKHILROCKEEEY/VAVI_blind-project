import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import functions
import yolopy
import speech
import cv2
import os
import detect
import datetime

# Import your functions, yolopy, speech, detect, and other necessary modules here

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "dfkey.json"

# Define CSS to set a background image
main_style = """
<style>
    body {
        background-color: blue;
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    .stApp {
        color: white;
    }
    input[type="text"], input[type="password"] {
        background-color: orange;
        color: white;
    }
</style>
"""
st.markdown(main_style, unsafe_allow_html=True)

def main():
    st.title("Jyoti: Virtual Assistant for the Visually Impaired")

    labelsPath = "yolo/coco.names"
    weightsPath = "yolo/yolov3.weights"
    configPath = "yolo/yolov3.cfg"
    args = {"threshold": 0.3, "confidence": 0.5}
    project_id = "vpi-model"

    engine = speech.speech_to_text()
    model = yolopy.yolo(labelsPath, weightsPath, configPath)
    listening = False
    intent = None

    cam = cv2.VideoCapture(0)

    # Streamlit UI
    if not listening:
        resp = st.text_input("Speak your command:")
        if st.button("Submit"):
            if resp:
                intent, text = detect.detect_intent_texts(project_id, 0, [resp], 'en')
            if intent == 'Jyoti' and resp:
                listening = True

    else:
        st.write("What can I help you with?")
        intent = ''
        st.write("Listening")
        resp = st.text_input("Speak your command:")
        st.write("Processing")
        if resp:
            intent, text = detect.detect_intent_texts(project_id, 0, [resp], 'en')
        if intent == 'Describe':
            st.write("Object Detection Results:")
            detected_labels = detect.describeScene(cam, model, engine)
            if detected_labels:
                for label, confidence in detected_labels.items():
                    accuracy_percentage = confidence * 100
                    st.write(f"Detected: {label} with accuracy: {accuracy_percentage:.2f}%")
        elif intent == 'endconvo':
            st.write(text)
            listening = False
            st.write(text)
        elif intent == 'Brightness':
            st.write("It is {} outside".format(functions.getBrightness(cam)[0]))
        elif intent == "FillForm":
            detect.detect_form(cam, engine)
        elif intent == "Read":
            st.write("Read text here")  # You can show the detected text or articles here
        elif intent == "Time":
            currentDT = datetime.datetime.now()
            st.write("The time is {} hours and {} minutes".format(currentDT.hour, currentDT.minute))
        elif resp != 'None':
            st.write(text)

if __name__ == "__main__":
    main()
