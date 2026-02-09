import json
import numpy as np
from typing import Dict

def load_poi_embeddings(file_path: str) -> Dict[str, np.ndarray]:
    """Load POI embeddings from JSON file."""
    with open(file_path, 'r') as f:
        embeddings_dict = json.load(f)
    
    # Ensure all embeddings are numpy arrays
    for poi_id in embeddings_dict:
        embeddings_dict[poi_id] = np.array(embeddings_dict[poi_id])
    
    return embeddings_dict

def variance_selection(embeddings_matrix: np.ndarray, target_dim: int) -> np.ndarray:
    """Select dimensions with highest variance across all POI embeddings."""
    variances = np.var(embeddings_matrix, axis=0)
    top_indices = np.argsort(variances)[-target_dim:]
    return top_indices

def compress_embeddings(embeddings_dict: Dict[str, np.ndarray], 
                       target_dim: int = 32) -> Dict[str, np.ndarray]:
    """Compress POI embeddings using variance selection method."""
    # Maintain original ID order for matrix conversion
    poi_ids = list(embeddings_dict.keys())
    embeddings_matrix = np.array([embeddings_dict[pid] for pid in poi_ids])
    
    print(f"Original dimension: {embeddings_matrix.shape[1]}")
    print(f"Compressing to: {target_dim}")
    
    selected_indices = variance_selection(embeddings_matrix, target_dim)
    
    compressed_dict = {}
    for i, poi_id in enumerate(poi_ids):
        # Keep ID as string for consistency with your loading logic
        compressed_dict[str(poi_id)] = embeddings_matrix[i, selected_indices]
    
    return compressed_dict

def save_compressed_embeddings(compressed_dict: Dict[str, np.ndarray], 
                              output_file: str):
    """Save compressed embeddings in MRS-compatible format."""
    with open(output_file, 'w') as f:
        f.write("{\n")
        items = list(compressed_dict.items())
        for i, (poi_id, embedding) in enumerate(items):
            embedding_str = "[" + ", ".join([f"{x:.4f}" for x in embedding]) + "]"
            comma = "," if i < len(items) - 1 else ""
            f.write(f'  "{poi_id}": {embedding_str}{comma}\n')
        f.write("}")

if __name__ == "__main__":
    INPUT_FILE = "poi_embeddings_selected.txt"
    OUTPUT_FILE = "poi_embeddings_compressed.txt"
    TARGET_DIM = 32  
    
    print("Loading 768-dim embeddings...")
    embeddings_dict = load_poi_embeddings(INPUT_FILE)
    
    print("Applying Variance Selection...")
    compressed_dict = compress_embeddings(embeddings_dict, target_dim=TARGET_DIM)
    
    save_compressed_embeddings(compressed_dict, OUTPUT_FILE)
    
    print(f"✓ Compression complete. File saved: {OUTPUT_FILE}")