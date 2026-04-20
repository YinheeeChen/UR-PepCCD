import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # 1) Load the full dataset.
    csv_path = "/workspace/guest/cyh/workspace/PepCCD/dataset/PepFlow/pepflow_dataset.csv"
    print(f"Loading full dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 2) Split with an 80/20 ratio. Keep random_state fixed for reproducibility.
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 3) Save train and test files.
    train_file = "/workspace/guest/cyh/workspace/PepCCD/dataset/PepFlow/pepflow_train.csv"
    test_file = "/workspace/guest/cyh/workspace/PepCCD/dataset/PepFlow/pepflow_test.csv"
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print("\nDataset split complete.")
    print(f"Total samples: {len(df)}")
    print(f"-> Train saved to {train_file}, count={len(train_df)}")
    print(f"-> Test  saved to {test_file}, count={len(test_df)}")

if __name__ == '__main__':
    main()
