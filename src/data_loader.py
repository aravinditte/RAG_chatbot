import pandas as pd

def load_data(file_path):
    
    data = pd.read_csv(file_path)
    print(f"Loaded {len(data)} records from {file_path}.")
    return data

if __name__ == "__main__":
    dataset = load_data("data\further_reduced_train.csv")
    print(dataset.head())
