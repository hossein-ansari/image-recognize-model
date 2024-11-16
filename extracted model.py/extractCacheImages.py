import os
import sys
import pickle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from pathlib import Path

def app(image_path, cache_file):
    weights_path = os.path.join("SLM", "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
    
    image_path = Path(image_path).resolve()
    cache_file = Path(cache_file).resolve()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        base_model = VGG16(weights=weights_path, include_top=False)
        model = Model(inputs=base_model.input, outputs=base_model.output)
    except Exception as e:
        print(f"Error loading VGG16 model: {e}")
        raise

    def extract_features(img_path):
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

    def load_cached_features(cache_file):
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache file: {e}")
                return [], [] 
        else:
            print(f"Cache file {cache_file} not found. Creating a new one.")
            return [], [] 

    def update_cache_with_new_image(image_path):
        features, file_names = load_cached_features(cache_file)
        new_features = extract_features(image_path)
        if new_features is not None:
            filename = os.path.basename(image_path)
            if filename not in file_names:
                features.append(new_features)
                file_names.append(filename)
                print(f"Cached features for {filename}.")
            else:
                print(f"Features for {filename} already cached, skipping...")
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump((features, file_names), f)
                print(f"Cache file updated: {cache_file}")
            except Exception as e:
                print(f"Error writing to cache file: {e}")
        else:
            print(f"Failed to extract features from {image_path}.")

    update_cache_with_new_image(image_path)

image_to_process = sys.argv[1]
cache_file = sys.argv[2]
app(image_to_process, cache_file)
