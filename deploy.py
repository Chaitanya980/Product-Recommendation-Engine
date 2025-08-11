import os
import subprocess

# Package model for TorchServe
os.system("torch-model-archiver --model-name recommendation --version 1.0 --serialized-file finetuned_model/pytorch_model.bin --handler transformers.handler --extra-files finetuned_model/config.json,finetuned_model/vocab.txt")

# Start TorchServe 
subprocess.run(["torchserve", "--start", "--model-store", "model_store", "--models", "recommendation=recommendation.mar"])

import requests
payload = {"text": "User history: prodA:Great product! ; Recommended next:"}
response = requests.post("http://localhost:8080/predictions/recommendation", json=payload)
print("Production inference:", response.json())
