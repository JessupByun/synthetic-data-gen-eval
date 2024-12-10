import pandas as pd

# Load the uploaded dataset
file_path = 'data/real_data/private_combined_df_cty_week2/private_combined_df_cty_week2_for_synth.csv'
insurance_data = pd.read_csv(file_path)

# Generate basic distribution metrics for each column
summary_stats = insurance_data.describe(include='all').transpose()

# Save to a file for easier viewing if necessary
summary_stats_file_path = 'evaluation/private_combined_df_cty_week2/private_combined_df_cty_week2_summary_stats.csv'
summary_stats.to_csv(summary_stats_file_path)

# Print the summary statistics to view directly in the console
print(summary_stats)

# To view the statistics as a CSV, you can now open the file:
print(f"Summary statistics saved to: {summary_stats_file_path}")
