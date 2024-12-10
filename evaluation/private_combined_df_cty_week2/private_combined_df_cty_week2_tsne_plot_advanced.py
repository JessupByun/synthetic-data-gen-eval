import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Fill in your file names here:
oracle_file = "data/real_data/private_combined_df_cty_week2/private_combined_df_cty_week2_for_synth.csv"        
train_file = "data/real_data/private_combined_df_cty_week2/private_combined_df_cty_week2_train.csv"       
llama_file = "data/synthetic_data/private_combined_df_cty_week2/combined_df_cty_week2_synthetic_data_llama70B_n200_temp1.0.csv"    
mixtral_file = "data/synthetic_data/private_combined_df_cty_week2/combined_df_cty_week2_synthetic_data_mixtral_n200_temp1.0_advanced_prompt.csv"

def load_and_encode_data(file_path):
    """Load CSV and apply one-hot encoding for categorical features."""
    df = pd.read_csv(file_path)
    # Convert categorical columns to dummy variables
    df = pd.get_dummies(df, drop_first=True)
    return df

# Load and encode data
df_oracle = load_and_encode_data(oracle_file)
df_train = load_and_encode_data(train_file)
df_llama = load_and_encode_data(llama_file)
df_mixtral = load_and_encode_data(mixtral_file)

# Suppose df_oracle is the "reference" set of columns
all_columns = set(df_oracle.columns).union(df_train.columns).union(df_llama.columns)

# Reindex each dataframe to have the same columns, filling missing ones with 0
df_oracle = df_oracle.reindex(columns=all_columns, fill_value=0)
df_train = df_train.reindex(columns=all_columns, fill_value=0)
df_llama = df_llama.reindex(columns=all_columns, fill_value=0)
df_mixtral = df_mixtral.reindex(columns=all_columns, fill_value=0)

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
# Adjust parameters as needed: 
tsne = TSNE(n_components=2, perplexity=20, random_state=42, max_iter=1000)
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
        marker_size = 100
    else:
        marker_size = 50

    plt.scatter(
        X_embedded[idx, 0], X_embedded[idx, 1],
        color=color_map[name],
        marker='o',
        s=marker_size,  # Add this line or change the size
        label=name,
        alpha=0.5
    )

plt.title("t-SNE Visualization: combined_df_cty_week2.csv n = 200 (max), temp = 1.0: Advanced Prompt")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()

# Set aspect to equal for a cleaner look
plt.gca().set_aspect('equal', 'box')

plt.tight_layout()
plt.show()
