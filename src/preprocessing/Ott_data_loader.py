import os
import pandas as pd

def load_ott_dataset(base_path):
    """Load the Ott et al. Opinion Spam dataset from text files."""
    
    data_path = os.path.join(base_path, "OpinionSpam", "Spam_Detection_Data")
    
    folders = {
        "deceptive_neg": {"label": "fake", "sentiment_label": "negative"},
        "deceptive_pos": {"label": "fake", "sentiment_label": "positive"},
        "truthful_neg": {"label": "genuine", "sentiment_label": "negative"},
        "truthful_pos": {"label": "genuine", "sentiment_label": "positive"},
    }
    
    reviews = []
    
    for folder, meta in folders.items():
        folder_path = os.path.join(data_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found, skipping.")
            continue
            
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                
                # Extract hotel name from filename (e.g., d_hilton_1.txt -> hilton)
                parts = filename.replace(".txt", "").split("_")
                hotel = parts[1] if len(parts) >= 3 else "unknown"
                
                reviews.append({
                    "text": text,
                    "label": meta["label"],
                    "sentiment_label": meta["sentiment_label"],
                    "hotel": hotel,
                    "source_folder": folder,
                    "filename": filename
                })
    
    df = pd.DataFrame(reviews)
    return df


def load_reallife_dataset(base_path):
    """Load the Real Life Trial dataset."""
    
    data_path = os.path.join(base_path, "RealLife", "Real_Life_Trial_Data")
    
    folders = {
        "Deceptive": "fake",
        "Truthful": "genuine"
    }
    
    reviews = []
    
    for folder, label in folders.items():
        folder_path = os.path.join(data_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found, skipping.")
            continue
            
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                
                reviews.append({
                    "text": text,
                    "label": label,
                    "source": "reallife",
                    "filename": filename
                })
    
    df = pd.DataFrame(reviews)
    return df


if __name__ == "__main__":
    base_path = os.path.join("data", "raw", "ott-dataset")
    
    # Load main dataset
    print("Loading Opinion Spam dataset...")
    df_main = load_ott_dataset(base_path)
    print(f"Loaded {len(df_main)} reviews")
    print(f"Label distribution:\n{df_main['label'].value_counts()}")
    print(f"Sentiment distribution:\n{df_main['sentiment_label'].value_counts()}")
    
    # Load real life dataset
    print("\nLoading Real Life dataset...")
    df_reallife = load_reallife_dataset(base_path)
    print(f"Loaded {len(df_reallife)} reviews")
    print(f"Label distribution:\n{df_reallife['label'].value_counts()}")
    
    # Save to processed folder
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    df_main.to_csv(os.path.join("data", "processed", "ott_reviews.csv"), index=False)
    df_reallife.to_csv(os.path.join("data", "processed", "reallife_reviews.csv"), index=False)
    
    print("\nDatasets saved to data/processed/")
    print(f"Main dataset sample:\n{df_main.head()}")
