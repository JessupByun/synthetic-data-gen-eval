import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Fill in your file names here:
oracle_file = "data/real_data/private_labMergeDemo1Trimmed/private_labMergeDemo1Trimmed.csv"        
train_file = "data/real_data/private_labMergeDemo1Trimmed/private_labMergeDemo1Trimmed_train.csv"           
llama_file = "data/synthetic_data/private_labMergeDemo1Trimmed/labMergeDemo1Trimmed_synthetic_data_llama70B_n250_temp1.0_advanced_prompt.csv"    
mixtral_file = "data/synthetic_data/private_labMergeDemo1Trimmed/labMergeDemo1Trimmed_synthetic_data_mixtral_n250_temp1.0_advanced_prompt.csv"

def load_and_encode_data(file_path):
    df = pd.read_csv(file_path)
    # If there are date/time columns that can't be numeric, consider dropping them or converting to a numeric representation first.
    # For example, if 'admittime' is a datetime column, you might need to parse and convert it to a numeric timestamp.
    # For now, try dropping any obvious date/time columns if they are not needed for the t-SNE:
    # df.drop(columns=["admittime_x", "dischtime", "charttime", ...], inplace=True, errors='ignore')
    
    df = pd.get_dummies(df, drop_first=True)
    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    # Fill NaNs
    df = df.fillna(0)
    return df

df_oracle = load_and_encode_data(oracle_file)
df_train = load_and_encode_data(train_file)
df_llama = load_and_encode_data(llama_file)
df_mixtral = load_and_encode_data(mixtral_file)

# Find union of columns
all_columns = set(df_oracle.columns) | set(df_train.columns) | set(df_llama.columns) | set(df_mixtral.columns)
all_columns = list(all_columns)

# Reindex and ensure numeric
def align_and_numeric(df):
    df = df.reindex(columns=all_columns, fill_value=0)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    return df

df_oracle = align_and_numeric(df_oracle)
df_train = align_and_numeric(df_train)
df_llama = align_and_numeric(df_llama)
df_mixtral = align_and_numeric(df_mixtral)

datasets = [
    ("Oracle", df_oracle),
    ("Train", df_train),
    ("LLaMa 70B", df_llama),
    ("Mixtral 8x7B", df_mixtral)
]

# Check for any object columns - print for debugging
for name, df in datasets:
    obj_cols = df.select_dtypes(include='object').columns
    if len(obj_cols) > 0:
        print(f"{name} has object columns: {obj_cols}")
        # Consider dropping or converting these columns properly

min_size = min(len(df) for _, df in datasets)
sampled_datasets = []
for name, df in datasets:
    df_sampled = df.sample(n=min_size, random_state=42)
    sampled_datasets.append((name, df_sampled))

all_data = []
all_labels = []
for name, df in sampled_datasets:
    all_data.append(df.values)
    all_labels.extend([name] * len(df))

X = np.vstack(all_data)

# Convert X to float explicitly
X = X.astype(float)

labels = np.array(all_labels)

# Optional: Double check for NaNs:
# Instead of using np.isnan(X).any(), we can rely on the conversion step.
# If conversion succeeded, we should be good.

tsne = TSNE(n_components=2, perplexity=25, random_state=42, max_iter=1000)
X_embedded = tsne.fit_transform(X)

color_map = {
    "Oracle": "cyan",
    "Train": "darkgrey",
    "LLaMa 70B": "red",
    "Mixtral 8x7B": "green"
}

plt.figure(figsize=(8, 6))

for name, _ in sampled_datasets:
    idx = (labels == name)
    if name in ["Oracle", "Train"]:
        marker_size = 100
    else:
        marker_size = 50

    plt.scatter(
        X_embedded[idx, 0], X_embedded[idx, 1],
        color=color_map[name],
        marker='o',
        s=marker_size,
        label=name,
        alpha=0.5
    )

plt.title("t-SNE Visualization: labMergeDemo1Trimmed.csv n=250, temp=1.0, Advanced Prompt")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.gca().set_aspect('equal', 'box')
plt.tight_layout()
plt.show()