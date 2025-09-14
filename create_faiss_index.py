import os
import csv
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def create_internship_text(internship):
    """Creates the text for embedding from the Title, Description, and Skills."""
    title = internship.get('Title', '')
    skills = internship.get('Skills', '')
    description = internship.get('Description', '')

    # Combine all three fields for a rich embedding
    return f"Internship Title: {title}. Description: {description}. Required skills: {skills}."


# 1. Load the Model
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
print(f"Loading model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")

# 2. Define file paths
ASSET_PATH = os.path.join('recommender', 'ml_assets')
CSV_PATH = os.path.join(ASSET_PATH, 'internships.csv')

# 3. Load NEW CSV and prepare texts
all_internships = []
# IMPORTANT: Verify these column names match your new CSV file
column_names = ['id', 'Title', 'Locations', 'Skills', 'Description']
with open(CSV_PATH, mode='r', encoding='utf-8') as csvfile:
    next(csvfile)  # Skip header
    reader = csv.DictReader(csvfile, fieldnames=column_names)
    for row in reader:
        all_internships.append(row)
print(f"Loaded {len(all_internships)} internships from CSV.")

internship_texts = [create_internship_text(internship) for internship in all_internships]

# 4. Generate and save the new FAISS Index and ID map
print("Generating new embeddings...")
all_embeddings = model.encode(internship_texts, convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(all_embeddings)
embedding_dimension = all_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dimension)
index.add(all_embeddings)
faiss.write_index(index, os.path.join(ASSET_PATH, 'internships.faiss'))

index_to_id_map = {i: internship['id'] for i, internship in enumerate(all_internships)}
with open(os.path.join(ASSET_PATH, 'index_to_id.pkl'), 'wb') as f:
    pickle.dump(index_to_id_map, f)

print("✅ New FAISS index and ID map saved successfully.")