import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from the .env file
load_dotenv()

# Access the Groq API key (should be contained in the .env)
api_key = os.getenv("GROQ_API_KEY")

# Instantiate the Groq client with API key
client = Groq(api_key=api_key)

# Model names
model_names = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "llama-3.2-1b-preview"]

# Data prompt template
prompt_template = """
Generate synthetic tabular data with the following structure:
- Column 1: Age (integer)
- Column 2: Income (float)
- Column 3: Gender (categorical: Male/Female)
- Column 4: Occupation (categorical: Engineer, Teacher, Doctor, Artist)
Please generate {num_rows} rows of data.
"""

# Function to generate synthetic data using a model and prompt
def generate_synthetic_data(model_name, num_rows):
    # Replace placeholders in prompt with specific values
    prompt = prompt_template.format(num_rows=num_rows)
    
    try:
        # Create a chat completion using the Groq API
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            model=model_name
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
        data = generate_synthetic_data(model_name, num_rows=5)

        print(f"Generated Data for {model_name}:\n{data}\n")

if __name__ == "__main__":
    main()