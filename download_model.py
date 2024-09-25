import kagglehub

# Download latest version
path = kagglehub.model_download("metaresearch/llama-3.1/transformers/8b")

print("Path to model files:", path)