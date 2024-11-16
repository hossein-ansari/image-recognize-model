import os
import sys
import pickle
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

def load_cached_features(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Cache file {cache_file} not found.")
        return None, None

def extract_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        return features.flatten()
    except Exception as e:
        print(f"Error extracting features from {img_path}: {e}")
        return None

def find_most_similar_image(input_image_path, cache_file):
    weights_path = "./SLM/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    base_model = VGG16(weights=weights_path, include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)
    features, file_names = load_cached_features(cache_file)
    if features is None or file_names is None:
        return
    input_features = extract_features(input_image_path, model)
    if input_features is None:
        return

    similarities = cosine_similarity([input_features], features)[0]
    
    threshold = 0.15
    similar_indices = np.where(similarities > threshold)[0]

    if len(similar_indices) == 0:
        print(f"No images found with similarity greater than {threshold * 100:.2f}%.")
        return
    most_similar_index = similar_indices[np.argmax(similarities[similar_indices])]
    most_similar_image = file_names[most_similar_index]
    # similarity_score = similarities[most_similar_index]

    print(most_similar_image)

input_image_path = sys.argv[1]
cache_file = sys.argv[2]
find_most_similar_image(input_image_path, cache_file)
