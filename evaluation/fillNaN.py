import pandas as pd

# Load the dataset
file_path = "data/synthetic_data/private_combined_df_cty_week2/combined_df_cty_week2_synthetic_data_mixtral_n200_temp1.0_advanced_prompt.csv"  # Replace with your dataset's file path
output_file = "data/synthetic_data/private_combined_df_cty_week2/combined_df_cty_week2_synthetic_data_mixtral_n200_temp1.0_advanced_prompt.csv"  # Replace with your desired output file name

# Read the dataset
df = pd.read_csv(file_path)

# Fill empty cells (NaN) with 0
df.fillna(0, inplace=True)

# Save the updated dataset
df.to_csv(output_file, index=False)

print(f"Updated dataset saved to {output_file}")
