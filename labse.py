import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import re


class LaBSEMatcher:
    def __init__(self, similarity_threshold=0.7):
       
        print("Loading LaBSE model...")
        self.model = SentenceTransformer('sentence-transformers/LaBSE')
        self.similarity_threshold = similarity_threshold
        print("Model loaded successfully!")
    
    def read_text_file(self, filepath):
        """Read text file and return content"""
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            return f.read()
    
   
    def split_into_paragraphs(self, text):
        """
        Reconstruct paragraphs for Malayalam PDF text
        Splits when strong sentence endings appear.
        """

        lines = text.splitlines()
        paragraphs = []
        current_para = ""

        for line in lines:
            line = line.strip()

            if not line:
                continue  # skip empty lines

            if current_para:
                current_para += " " + line
            else:
                current_para = line

            # If line ends with strong punctuation → treat as paragraph end
            if line.endswith(('.', '?', '!', '”', '"')):
                paragraphs.append(current_para.strip())
                current_para = ""

        # Add remaining text
        if current_para:
            paragraphs.append(current_para.strip())

        return paragraphs
        
    def get_embeddings(self, paragraphs):
        """Generate LaBSE embeddings for list of paragraphs"""
        print(f"Generating embeddings for {len(paragraphs)} paragraphs...")
        embeddings = self.model.encode(paragraphs, convert_to_numpy=True)
        return embeddings
    
    def find_similar_paragraphs(self, en_paragraphs, ml_paragraphs):

        print("Computing embeddings for English paragraphs...")
        en_embeddings = self.get_embeddings(en_paragraphs)
        
        print("Computing embeddings for Malayalam paragraphs...")
        ml_embeddings = self.get_embeddings(ml_paragraphs)
        
        print("Computing similarity matrix...")
        similarity_matrix = cosine_similarity(en_embeddings, ml_embeddings)
        
        matched_pairs = []
        used_ml_indices = set()
        
        # For each English paragraph, find the best matching Malayalam paragraph
        for en_idx in range(len(en_paragraphs)):
            # Find the best match for this English paragraph
            ml_idx = np.argmax(similarity_matrix[en_idx])
            max_similarity = similarity_matrix[en_idx][ml_idx]
            
            # Check if similarity exceeds threshold and ML paragraph hasn't been used
            if max_similarity >= self.similarity_threshold and ml_idx not in used_ml_indices:
                matched_pairs.append({
                    'en_text': en_paragraphs[en_idx],
                    'ml_text': ml_paragraphs[ml_idx],
                    'similarity': float(max_similarity),
                    'en_index': en_idx,
                    'ml_index': ml_idx
                })
                used_ml_indices.add(ml_idx)
        
        print(f"Found {len(matched_pairs)} matching paragraph pairs!")
        return matched_pairs


def save_to_csv(matched_pairs, output_file='dataset.csv'):
    """Append matched pairs to dataset.csv"""

    file_exists = os.path.isfile(output_file)

    with open(output_file, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)

        # Write header only if file doesn't exist
        if not file_exists:
            writer.writerow(['English Text (en_xx)', 'Malayalam Text (ml_xx)'])

        # Append rows
        for pair in matched_pairs:
            writer.writerow([
                f"en_xx: {pair['en_text']}",
                f"ml_xx: {pair['ml_text']}",
            ])

    print(f"Appended {len(matched_pairs)} matches to {output_file}")

def main():
    """Main execution function"""
    
    # Configuration
    EN_TEXT_FILE = 'eng.txt'  # Path to English text file
    ML_TEXT_FILE = 'mal.txt'  # Path to Malayalam text file
    OUTPUT_CSV = 'matched_paragraphs.csv'  # Output CSV file
    SIMILARITY_THRESHOLD = 0.7  # Adjust based on your needs (0.0 to 1.0)
    
    # Step 1: Initialize LaBSE matcher
    matcher = LaBSEMatcher(similarity_threshold=SIMILARITY_THRESHOLD)
    
    # Step 2: Read text files
    print("\nReading text files...")
    en_text = matcher.read_text_file(EN_TEXT_FILE)
    ml_text = matcher.read_text_file(ML_TEXT_FILE)
    
    # Step 3: Split into paragraphs
    print("\nSplitting texts into paragraphs...")
    en_paragraphs = matcher.split_into_paragraphs(en_text)
    ml_paragraphs = matcher.split_into_paragraphs(ml_text)
    print(f"English: {len(en_paragraphs)} paragraphs")
    print(f"Malayalam: {len(ml_paragraphs)} paragraphs")
    
    # Step 4: Find similar paragraphs
    print("\nFinding similar paragraphs...")
    matched_pairs = matcher.find_similar_paragraphs(en_paragraphs, ml_paragraphs)
    
    # Step 5: Save to CSV
    print("\nSaving to CSV...")
    save_to_csv(matched_pairs, OUTPUT_CSV)
    
    print("\nProcess completed successfully!")
    print(f"Total matches found: {len(matched_pairs)}")
    print(f"Check output file: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
