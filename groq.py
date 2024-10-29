import groq

# Make as env variable for more security in future
api_key = ""

# Assuming Groq API has a function for model loading
model_name = "gpt-neox-20b"
model = groq.load_model(model_name)

# Prepare input data (e.g., a text prompt for language models)
input_text = "What is the capital of France?"
input_data = {"text": input_text}

# Run the model inference using Groq API
response = model.run(input_data)

# Process and print the output
output_text = response.get("generated_text", "No output")
print(f"Generated text: {output_text}")