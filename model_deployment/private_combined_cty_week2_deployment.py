import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from groq import Groq

# Load the real data
data_csv = "data/real_data/private_combined_df_cty_week2/private_combined_df_cty_week2_for_synth.csv"
data = pd.read_csv(data_csv)

# Split the data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.8, random_state=42)
test_data.to_csv('data/real_data/private_combined_df_cty_week2/private_combined_df_cty_week2_test.csv', index=True)
train_data.to_csv('data/real_data/private_combined_df_cty_week2/private_combined_df_cty_week2_train.csv', index=False) # Will include the entire training data, which will then be sampled in n sample sizes below.

# Define the n sample size of train_data
train_data = train_data.sample(50)

# Define temperature parameter for model (controls randomness and diversity, as temp -> 0, model becomes more deterministic and repetitive)
temperature = 1

# Load environment variables from the .env file
load_dotenv()

# Access the Groq API key (should be securely stored in the .env file (not provided, must be generated by user))
api_key = os.getenv("GROQ_API_KEY")

# Instantiate the Groq client with API key
client = Groq(api_key=api_key)

# List of model ID names that will be deployed. Visit groq API documentation for more models
model_names = ["llama-3.1-70b-versatile"] #, "llama-3.1-8b-instant", "llama-3.2-1b-preview"]

# This prompt structure is adapted from the prompt example B.5. from the research paper: "Curated LLM: Synergy of LLMs and Data Curation for tabular augmentation in low-data regimes" (Seedatk, Huynh, et al.) https://arxiv.org/pdf/2312.12112 
# The template is currently adapted to the 'insurance.csv' dataset (referenced in README.md)
prompt_template = """
System role: You are a tabular synthetic data generation model.

You are a synthetic data generator.
Your goal is to produce data which mirrors the given examples in causal structure and feature and label distributions but also produce as diverse samples as possible.

I will give you real examples first.

Context: Leverage your knowledge about COVID-19 trends, social behavior, and population statistics to generate 1000 realistic but diverse samples.
Output the data in a csv format where I can directly copy and paste into a csv.

Example data: {data}

The output should use the following schema:

"state": string, // feature column for the state
"county_fips": string, // feature column for the county FIPS code
"week": string, // feature column for the week range
"mask_user_pct": float, // feature column for the percentage of mask users
"mask_mandate": string, // feature column, yes or no for mask mandate
"gop_vote_share_2016": float, // feature column for GOP vote share in 2016
"deaths_per_10k": float, // feature column for deaths per 10,000 (nullable)
"COVID_news": float, // feature column for COVID news coverage percentage
"retail_visit_per_hundred": float, // feature column for retail visits per 100 people
"COVID_news_cable": float, // feature column for COVID-related cable news coverage
"urban_population_percentage": float, // feature column for urban population percentage
"image_users": integer, // feature column for number of image users
"mask_users": integer, // feature column for number of mask users
"population_density": float, // feature column for population density
"week_counter": integer, // feature column for week counter
"week_counter_log": float, // feature column for log of week counter
"population": integer // feature column for total population

DO NOT COPY THE EXAMPLES but generate realistic but new and diverse samples which have the correct label conditioned on the features.
"""

# Function to generate synthetic data using a model and prompt
def generate_synthetic_data(model_name, data):

    prompt = prompt_template.format(data = data)
    
    try:
        # Create a chat completion using the Groq API
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            model=model_name,
            #response_format={"type": "json_object"} Turn on for JSON beta mode
            temperature=temperature
        )
        
        # Print the full response for debugging
        print("Full Response:", response)
        
        generated_data = response.choices[0].message.content if response.choices else "No output"
        
        return generated_data
    except Exception as e:
        print(f"Error generating data with model {model_name}: {e}")
        return None

# Main function to run the process
def main():
    
    for model_name in model_names:
        print(f"Generating data with {model_name}...")
        
        # Generate synthetic data with n rows!
        data = generate_synthetic_data(model_name, train_data)

        print(f"Generated Data for {model_name}:\n{data}\n")

if __name__ == "__main__":
    main()