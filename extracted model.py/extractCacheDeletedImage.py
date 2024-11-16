import os
import pickle
from pathlib import Path
import sys
def delete_image_from_cache(image_path, cache_file):
    cache_file = Path(cache_file).resolve()

    def load_cached_features(cache_file):
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache file: {e}")
                return [], [] 
        else:
            print(f"Cache file {cache_file} not found.")
            return [], [] 

    def save_cache(features, file_names, cache_file):
        try:
            with open(cache_file, "wb") as f:
                pickle.dump((features, file_names), f)
            print(f"Cache file updated: {cache_file}")
        except Exception as e:
            print(f"Error writing to cache file: {e}")

    features, file_names = load_cached_features(cache_file)
    filename = os.path.basename(image_path)

    if filename in file_names:
        index = file_names.index(filename)
        del file_names[index]
        del features[index]
        print(f"Deleted cached data for {filename}.")
        save_cache(features, file_names, cache_file)
    else:
        print(f"No cached data found for {filename}. Skipping deletion.")

image_to_delete = sys.argv[1]
cache_file = sys.argv[2]
delete_image_from_cache(image_to_delete, cache_file)
