# Person Clustering & Retrieval System Documentation

## Introduction
This project implements a **Person Clustering & Retrieval System** using deep learning-based feature extraction, unsupervised clustering (HDBSCAN), and FAISS-based nearest neighbor search. The system allows users to:
- Extract features from images using MobileNetV2.
- Perform clustering on extracted features using HDBSCAN.
- Retrieve similar images using FAISS-based nearest neighbor search.

## Dependencies
Ensure you have the following Python libraries installed before running the application:

```bash
pip install numpy opencv-python streamlit pillow hdbscan faiss-cpu tensorflow scikit-learn pickle-mixin
```

## Folder Structure
```
project/
│-- sampledata/          # Folder containing dataset images
│-- README.md            # Project documentation
│-- app.py               # Main Streamlit application
│-- features.pkl         # Pickled file storing extracted features and filenames
```

## Components Overview

### 1. Image Loading
Images are loaded from the `sampledata` folder and resized to **224x224 pixels**.

```python
def load_images_from_folder(folder, img_size=(224, 224)):
    images, filenames = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            filenames.append(filename)
    return images, filenames
```

### 2. Feature Extraction using MobileNetV2
MobileNetV2 is used for extracting deep features from images.

```python
def extract_feature(img, model):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = model.predict(img_array).flatten()
    return feature
```

### 3. Parallelized Feature Extraction
To speed up processing, `ThreadPoolExecutor` is used to extract features in parallel.

```python
def extract_features_parallel(images, model):
    with ThreadPoolExecutor() as executor:
        features = list(executor.map(lambda img: extract_feature(img, model), images))
    return np.array(features)
```

### 4. Saving and Loading Features
Features are stored in a `.pkl` file for reuse.

```python
def save_features(features, filenames, file_path="features.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump((features, filenames), f)

def load_features(file_path="features.pkl"):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None, None
```

### 5. Clustering using HDBSCAN
HDBSCAN is used for unsupervised clustering of image embeddings.

```python
def perform_clustering(features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    pca = PCA(n_components=50)
    features_reduced = pca.fit_transform(features_scaled)
    
    clusterer = HDBSCAN(min_cluster_size=3, metric="euclidean")
    labels = clusterer.fit_predict(features_reduced)
    return labels
```

### 6. FAISS-based Image Retrieval
FAISS is used for efficient nearest neighbor search.

```python
def build_faiss_index(features):
    features = np.array(features, dtype="float32")
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    return index
```

### 7. Searching Similar Images
Given a query image, similar images are retrieved from FAISS.

```python
def find_similar_images_faiss(input_image, index, features, filenames, model, images, top_n=10):
    input_image = input_image.resize((224, 224))
    img_array = image.img_to_array(input_image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    query_feature = model.predict(img_array).flatten().astype("float32")

    _, indices = index.search(np.array([query_feature]), top_n)
    similar_images = [(filenames[idx], images[idx]) for idx in indices[0]]
    return similar_images
```

## Running the Application
Run the Streamlit app using:
```bash
streamlit run app.py
```

## Streamlit Interface
The Streamlit app provides:
1. **Feature Extraction**: Extracts features from dataset images.
2. **Clustering Visualization**: Displays clustering results.
3. **Image Retrieval**: Allows users to upload an image and retrieve similar images.

## Expected Output
- Clustered images with cluster labels.
- Similar images retrieved using FAISS.
- Query image assigned to a specific cluster.

## Conclusion
This system provides an end-to-end pipeline for person clustering and image retrieval. Future improvements can include:
- Using a different deep learning model for feature extraction.
- Fine-tuning HDBSCAN parameters.
- Adding a UI for cluster visualization.

