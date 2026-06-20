import os
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

# Load environmental configs from local workspace paths
load_dotenv()  

try:
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") or ""
    print("✅ Hugging Face token loaded successfully!")
except Exception as e:
    print("ℹ️ HF_TOKEN not found in environment, proceeding with public access models.")

# 1. Load CodeBERT with the safetensors fix
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, use_safetensors=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"🚀 CodeBERT is ready on device: {device}\n")

# ----------------------------------------------------------------------
# 2. READ AND ANALYZE THE TARGET SYSTEM FILE
# ----------------------------------------------------------------------

# Relative file path pointing into your FastAPI system layout modules
target_file_path = "routes/llm.py"

if not os.path.exists(target_file_path):
    print(f"❌ Error: Could not find '{target_file_path}'. Make sure the path is correct.")
else:
    # Read the raw code file contents securely from stream reader configurations
    with open(target_file_path, "r", encoding="utf-8") as f:
        raw_code = f.read()

    print(f"📋 [Analyzing File: {target_file_path}]")
    print("-" * 65)

    # Metric: Split out separate line strings to calculate basic sizing properties
    lines = raw_code.splitlines()
    print(f"🔹 Total Lines of Code: {len(lines)}")

    # Metric: Find and parse out Python method and function definitions dynamically 
    functions_found = re.findall(r"def\s+(\w+)\s*\(", raw_code)
    print(f"🔹 Total Functions Found: {len(functions_found)}")
    for func in functions_found:
        print(f"  - {func}()")

    # Metric: Isolate decorators targeting router connections to flag network nodes
    endpoints_found = re.findall(r"@router\.(post|get|put|delete)\s*\(", raw_code)
    print(f"🔹 Total FastAPI Routes Found: {len(endpoints_found)}")
    print("-" * 65)

    # ----------------------------------------------------------------------
    # 3. USE CODEBERT TO COMPARE THE FIRST TWO DISCOVERED FUNCTIONS
    # ----------------------------------------------------------------------
    
    def get_embedding(code_text):
        """Converts arbitrary raw text data maps into dense tensor representations."""
        inputs = tokenizer(code_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    print("\n🧠 Extracting and comparing internal functions using CodeBERT...")
    
    # Verify that the array has a sufficient balance of target items to perform similarity metrics
    if len(functions_found) >= 2:
        try:
            name_1 = functions_found[0]
            name_2 = functions_found[1]

            # Slice out the function body blocks cleanly using dynamic string indexing
            func_1_code = raw_code.split(f"def {name_1}")[1].split("def ")[0]
            func_2_code = raw_code.split(f"def {name_2}")[1].split("def ")[0]

            # Generate mathematical vector distributions
            vector_1 = get_embedding(func_1_code)
            vector_2 = get_embedding(func_2_code)

            # Measure contextual/logical overlaps using directional tracking scores
            similarity = F.cosine_similarity(vector_1, vector_2)

            print("=" * 65)
            print(f"📊 CodeBERT Similarity between [{name_1}()] and [{name_2}()]:")
            print(f"   Score: {similarity.item():.4f}")
            print("=" * 65)
            
        except Exception as e:
            print(f"ℹ️ Could not automatically slice matching functional scopes: {str(e)}")
    else:
        print("ℹ️ Need at least 2 internal functions inside the file to perform a CodeBERT comparison.")