import os
import cv2
import numpy as np
import streamlit as st
from io import BytesIO
from PIL import Image
import pickle
from concurrent.futures import ThreadPoolExecutor

#HDBSCAN
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import faiss

# Deep Learning 
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image


DATASET_FOLDER = "sampledata"  
FEATURES_FILE = "features.pkl"
IMG_SIZE = (224, 224)
TOP_N = 10  


def load_images_from_folder(folder, img_size=IMG_SIZE):
    images, filenames = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            filenames.append(filename)
    return images, filenames

def save_features(features, filenames, file_path=FEATURES_FILE):
    with open(file_path, "wb") as f:
        pickle.dump((features, filenames), f)

def load_features(file_path=FEATURES_FILE):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None, None

def extract_feature(img, model):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = model.predict(img_array).flatten()
    return feature

def extract_features_parallel(images, model):
    with ThreadPoolExecutor() as executor:
        features = list(executor.map(lambda img: extract_feature(img, model), images))
    return np.array(features)

def build_faiss_index(features):
    features = np.array(features, dtype="float32")
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    return index

def find_similar_images_faiss(input_image, index, features, filenames, model, images, top_n=TOP_N):
    input_image = input_image.resize(IMG_SIZE)
    img_array = image.img_to_array(input_image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    query_feature = model.predict(img_array).flatten().astype("float32")

    # Here Search in FAISS index
    _, indices = index.search(np.array([query_feature]), top_n)
    similar_images = [(filenames[idx], images[idx]) for idx in indices[0]]
    return similar_images

def perform_clustering(features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    pca = PCA(n_components=50)
    features_reduced = pca.fit_transform(features_scaled)
    
    clusterer = HDBSCAN(min_cluster_size=3, metric="euclidean")
    labels = clusterer.fit_predict(features_reduced)
    return labels

# --- MAIN APPLICATION USING STREAMLIT ---

st.title("Person Clustering & Retrieval System")

# Here i have Load the MobileNetV2 model
st.info("Loading deep model...")
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')


features, filenames = load_features()
if features is None:
    st.info("Extracting features from dataset images. Please wait...")
    images, filenames = load_images_from_folder(DATASET_FOLDER)
    features = extract_features_parallel(images, model)
    save_features(features, filenames)
else:
    # Here i have need load images if features are precomputed 
    images, filenames = load_images_from_folder(DATASET_FOLDER)

st.success("Feature extraction complete!")

# Clustering Stage
st.subheader("Clustering")
st.info("Clustering images with HDBSCAN...")
labels = perform_clustering(features)
st.write("Clustering Results:")
for label in np.unique(labels):
    st.write(f"Cluster {label}: {np.sum(labels == label)} images")

# Here I have Build FAISS index for retrieval
faiss_index = build_faiss_index(features)


st.subheader("Search & Retrieve")
uploaded_file = st.file_uploader("Upload a query image (person)", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Display uploaded image
    image_bytes = uploaded_file.read()
    query_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    st.image(query_img, caption="Query Image", use_container_width=True)
    
    # Similar images have Retrieve using FAISS
    similar_images = find_similar_images_faiss(query_img, faiss_index, features, filenames, model, images)
    
    st.subheader("Retrieved Images:")
    cols = st.columns(5)
    for i, (fname, img) in enumerate(similar_images):
        with cols[i % 5]:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=fname, use_container_width=True)
    
    
    query_feature = extract_feature(np.array(query_img.resize(IMG_SIZE)), model)
    query_feature = query_feature.astype("float32").reshape(1, -1)
    _, indices = faiss_index.search(query_feature, 1)
    query_label = labels[indices[0][0]]
    st.info(f"The query image is likely in cluster: {query_label}")
