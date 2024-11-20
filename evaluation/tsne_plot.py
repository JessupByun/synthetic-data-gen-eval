import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load data
oracle_data = pd.read_csv('data/real_data/insurance.csv')  # The entire real dataset
train_data = pd.read_csv('data/real_data/insurance_train.csv')  # Subset of oracle data used for training

# Manually specify the paths to the two synthetic data samples
synthetic_data_sample_1 = pd.read_csv('data/synthetic_data/insurance_synthetic_data_llama70B_n40_temp0.3.csv')
synthetic_data_sample_2 = pd.read_csv('data/synthetic_data/insurance_synthetic_data_llama70B_n100_temp0.7.csv')

# Sample N points from each dataset
N = 100
oracle_sample = oracle_data.sample(N, random_state=42)
train_sample = train_data.sample(N, random_state=42)
synthetic_sample_1 = synthetic_data_sample_1.sample(N, random_state=42)
synthetic_sample_2 = synthetic_data_sample_2.sample(N, random_state=42)

# Concatenate all datasets and create labels
all_data = pd.concat([oracle_sample, train_sample, synthetic_sample_1, synthetic_sample_2])
labels = (['Oracle'] * N) + (['Train'] * N) + (['LLaMA 70B-n40-temp0.3'] * N) + (['LLaMA 70B-n100-temp0.7'] * N)

# Standardize and apply t-SNE with adjusted perplexity for a more compact shape
numeric_data = all_data.select_dtypes(include=np.number)
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
tsne_results = tsne.fit_transform(numeric_data)

# Plotting with custom colors
plt.figure(figsize=(10, 6))
color_map = {
    'Oracle': 'orange',
    'Train': 'dimgray',
    'LLaMA 70B-n40-temp0.3': 'dodgerblue',
    'LLaMA 70B-n100-temp0.7': 'crimson'
}

for label in set(labels):
    indices = [i for i, lbl in enumerate(labels) if lbl == label]
    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=label, color=color_map[label], alpha=0.6)

plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Plot of Oracle, Train, and Selected Synthetic Data Samples')
plt.legend()
plt.show()
