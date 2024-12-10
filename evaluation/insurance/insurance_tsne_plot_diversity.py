import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Fill in your file names here:
oracle_file = "data/real_data/insurance/insurance.csv"
train_file = "data/real_data/insurance/insurance_train.csv"

# Four LLaMA synthetic datasets at different temperatures:
llama_file_temp03 = "data/synthetic_data/insurance/insurance_synthetic_data_llama70B_n250_temp0.3.csv"
llama_file_temp05 = "data/synthetic_data/insurance/insurance_synthetic_data_llama70B_n250_temp0.5.csv"
llama_file_temp07 = "data/synthetic_data/insurance/insurance_synthetic_data_llama70B_n250_temp0.7.csv"
llama_file_temp1 = "data/synthetic_data/insurance/insurance_synthetic_data_llama70B_n250_temp1.0.csv"

def load_and_encode_data(file_path):
    """Load CSV and apply one-hot encoding for categorical features."""
    df = pd.read_csv(file_path)
    # Convert categorical columns to dummy variables
    df = pd.get_dummies(df, drop_first=True)
    return df

# Load and encode data
df_oracle = load_and_encode_data(oracle_file)
df_train = load_and_encode_data(train_file)

df_llama_03 = load_and_encode_data(llama_file_temp03)
df_llama_05 = load_and_encode_data(llama_file_temp05)
df_llama_07 = load_and_encode_data(llama_file_temp07)
df_llama_1 = load_and_encode_data(llama_file_temp1)

# Combine into a list of (name, dataframe)
datasets = [
    ("Oracle", df_oracle),
    ("Train", df_train),
    ("LLaMa 3.1 70B temp = 0.3", df_llama_03),
    ("LLaMa 3.1 70B temp = 0.5", df_llama_05),
    ("LLaMa 3.1 70B temp = 0.7", df_llama_07),
    ("LLaMa 3.1 70B temp = 1.0", df_llama_1)
]

# Find the union of all columns across datasets
all_columns = set()
for _, df in datasets:
    all_columns.update(df.columns)
all_columns = list(all_columns)

# Reindex each dataset to have the same columns, filling missing ones with 0
aligned_datasets = []
for name, df in datasets:
    df_aligned = df.reindex(columns=all_columns, fill_value=0)
    aligned_datasets.append((name, df_aligned))

# Determine the smallest dataset size (just in case sizes differ)
min_size = min(len(df) for _, df in aligned_datasets)

# Sample each dataset to the smallest size to ensure equal representation
sampled_datasets = []
for name, df in aligned_datasets:
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
tsne = TSNE(n_components=2, perplexity=20, random_state=42, max_iter=1000)
X_embedded = tsne.fit_transform(X)

# Define custom colors for each dataset
color_map = {
    "Oracle": "cyan",
    "Train": "darkgrey",
    "LLaMa 3.1 70B temp = 0.3": "blue",
    "LLaMa 3.1 70B temp = 0.5": "red",
    "LLaMa 3.1 70B temp = 0.7": "orange",
    "LLaMa 3.1 70B temp = 1.0": "green"
}

plt.figure(figsize=(10, 8))

for name, _ in sampled_datasets:
    idx = (labels == name)

    # Set a larger size for Oracle and Train to highlight them
    if name in ["Oracle", "Train"]:
        marker_size = 200
        transparency = 1
    else:
        marker_size = 70
        transparency = 0.5

    plt.scatter(
        X_embedded[idx, 0], X_embedded[idx, 1],
        color=color_map[name],
        marker='o',
        s=marker_size,
        label=name,
        alpha=transparency,
        linewidths=0.5
    )

plt.title("t-SNE Visualization of Insurance Data: Diversity by LLaMa Temperature (n=250)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()

# Set aspect to equal for a cleaner look
plt.gca().set_aspect('equal', 'box')

plt.tight_layout()
plt.show()
