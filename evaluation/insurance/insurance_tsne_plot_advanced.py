import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# File paths
oracle_file = "data/real_data/insurance/insurance.csv"
train_file = "data/real_data/insurance/insurance_train.csv"
llama_file = "data/synthetic_data/insurance/insurance_synthetic_data_llama70B_n250_temp1.0_advanced_prompt.csv"
mixtral_file = "data/synthetic_data/insurance/insurance_synthetic_data_mixtral_n250_temp1.0_advanced_prompt.csv"

def load_and_encode_data(file_path, reference_columns=None):
    """Load CSV and apply one-hot encoding for categorical features."""
    df = pd.read_csv(file_path)
    df = pd.get_dummies(df, drop_first=True)
    if reference_columns is not None:
        # Ensure the same columns as the reference
        for col in reference_columns:
            if col not in df:
                df[col] = 0
        df = df[reference_columns]  # Match column order
    return df

# Load the reference dataset first to get consistent columns
reference_df = pd.read_csv(oracle_file)
reference_columns = pd.get_dummies(reference_df, drop_first=True).columns

# Load and encode data
df_oracle = load_and_encode_data(oracle_file, reference_columns)
df_train = load_and_encode_data(train_file, reference_columns)
df_llama = load_and_encode_data(llama_file, reference_columns)
df_mixtral = load_and_encode_data(mixtral_file, reference_columns)

# Combine into a list of (name, dataframe)
datasets = [
    ("Oracle", df_oracle),
    ("Train", df_train),
    ("LLaMa 70B", df_llama),
    ("Mixtral 8x7B", df_mixtral)
]

# Determine the smallest dataset size
min_size = min(len(df) for _, df in datasets)

# Sample each dataset to the smallest size to ensure equal representation
sampled_datasets = []
for name, df in datasets:
    df_sampled = df.sample(n=min_size, random_state=42)
    sampled_datasets.append((name, df_sampled))

# Extract features and labels
all_data = []
all_labels = []
for name, df in sampled_datasets:
    all_data.append(df.values)
    all_labels.extend([name] * len(df))

X = np.vstack(all_data)
labels = np.array(all_labels)

# Run t-SNE on the combined data
tsne = TSNE(n_components=2, perplexity=15, random_state=42, max_iter=1000)
X_embedded = tsne.fit_transform(X)

# Define custom colors
color_map = {
    "Oracle": "cyan",
    "Train": "darkgrey",
    "LLaMa 70B": "red",
    "Mixtral 8x7B": "green"
}

# Plot the results
plt.figure(figsize=(8, 6))

for name, _ in sampled_datasets:
    idx = (labels == name)
    
    # Set a larger size for Oracle and Train
    if name in ["Oracle", "Train"]:
        marker_size = 150
        transparency = 0.3
    else:
        marker_size = 70
        transparency = 0.5

    plt.scatter(
        X_embedded[idx, 0], X_embedded[idx, 1],
        color=color_map[name],
        marker='o',
        s=marker_size,  # Add this line or change the size
        label=name,
        alpha=transparency
    )

plt.title("t-SNE Visualization: insurance.csv n ~ 250, temp = 1.0: Advanced Prompt")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()

# Set aspect to equal for a cleaner look
plt.gca().set_aspect('equal', 'box')

plt.tight_layout()
plt.show()
