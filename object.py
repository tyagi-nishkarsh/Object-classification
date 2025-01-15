import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import scipy.io.wavfile as wavfile
import tempfile
from gtts import gTTS


# Initialize pipelines
narrator = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")
object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")

def generate_audio(text):
    """Generate audio from text."""
    tts = gTTS(text)
    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(temp_audio_file.name)
    return temp_audio_file.name


def read_objects(detection_objects):
    """Generate natural text describing detected objects."""
    object_counts = {}
    for detection in detection_objects:
        label = detection['label']
        object_counts[label] = object_counts.get(label, 0) + 1

    response = "This picture contains"
    labels = list(object_counts.keys())
    for i, label in enumerate(labels):
        response += f" {object_counts[label]} {label}"
        if object_counts[label] > 1:
            response += "s"
        if i < len(labels) - 2:
            response += ","
        elif i == len(labels) - 2:
            response += " and"
    response += "."
    return response


def draw_bounding_boxes(image, detections, font_size=20):
    """Draw bounding boxes on the image."""
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for detection in detections:
        box = detection['box']
        xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)

        label = detection['label']
        score = detection['score']
        text = f"{label} {score:.2f}"

        text_size = draw.textbbox((xmin, ymin), text, font=font)
        draw.rectangle([(text_size[0], text_size[1]), (text_size[2], text_size[3])], fill="red")
        draw.text((xmin, ymin), text, fill="white", font=font)

    return draw_image


def detect_objects_in_image(image):
    """Detect objects, process image, and generate audio."""
    detections = object_detector(image)
    processed_image = draw_bounding_boxes(image, detections)
    natural_text = read_objects(detections)
    audio_file = generate_audio(natural_text)
    return processed_image, natural_text, audio_file


# Streamlit UI
st.title("@GenAILearniverse Project 7: Object Detector with Audio")
st.write("This application highlights objects and generates an audio description for the input image.")

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Objects"):
        with st.spinner("Processing..."):
            processed_image, description, audio_path = detect_objects_in_image(image)

            st.image(processed_image, caption="Processed Image", use_column_width=True)
            st.text_area("Object Descriptions", description, height=100)
            st.audio(audio_path)
