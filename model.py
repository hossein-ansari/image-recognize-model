import os
import numpy as np
import pickle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import sys

def app():
    weights_path = "./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
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

    def cache_reference_features(directory_path, cache_file="reference_features.pkl"):
        features_list = []
        file_names = []
        try:
            for filename in os.listdir(directory_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(directory_path, filename)
                    features = extract_features(img_path)
                    if features is not None:
                        features_list.append(features)
                        file_names.append(filename)
            with open(cache_file, "wb") as f:
                pickle.dump((features_list, file_names), f)
        except Exception as e:
            print(f"Error accessing directory {directory_path}: {e}")
        return features_list, file_names

    def load_cached_features(cache_file="reference_features.pkl"):
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        else:
            print(f"Cache file {cache_file} not found.")
            return None, None

    def find_most_similar_image(query_features, reference_features):
        try:
            similarities = cosine_similarity([query_features], reference_features)
            return similarities[0]
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return None

    reference_images_directory = "./images"
    cache_file = "reference_features.pkl"

    reference_features, reference_image_names = load_cached_features(cache_file)
    if reference_features is None:
        reference_features, reference_image_names = cache_reference_features(
            reference_images_directory, cache_file
        )

    query_image_path = sys.argv[1]
    query_features = extract_features(query_image_path)

    if query_features is not None and len(reference_features) > 0:
        similarity_scores = find_most_similar_image(query_features, reference_features)
        if similarity_scores is not None:
            most_similar_index = np.argmax(similarity_scores)
            most_similar_image_name = reference_image_names[most_similar_index]
            most_similar_score = similarity_scores[most_similar_index]

            if most_similar_score > 0.15:
                print(f"The most similar image is: {most_similar_image_name}")
                print(f"Similarity score: {most_similar_score}")
            else:
                print(
                    f"No image meets the similarity threshold of 15%. {most_similar_score}"
                )
        else:
            print("Error: Unable to determine the similarity scores.")
    else:
        print("Error: Unable to extract features from query image or reference images.")
app()
