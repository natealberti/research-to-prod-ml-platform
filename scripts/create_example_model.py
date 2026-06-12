import json
from pathlib import Path

import torch
from torchvision.models import resnet18


OUTPUT_DIR = Path("examples/resnet18_classifier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# create model
model = resnet18(num_classes=10)
model.eval()

# save checkpoint
torch.save(model.state_dict(), OUTPUT_DIR / "checkpoint.pt")

# create sample input
sample_input = torch.randn(1, 3, 224, 224)
torch.save(sample_input, OUTPUT_DIR / "sample_input.pt")

# run inference
with torch.no_grad():
    output = model(sample_input)

# save expected output metadata
expected_output = {
    "shape": list(output.shape),
    "dtype": str(output.dtype),
}

with open(OUTPUT_DIR / "expected_output.json", "w") as f:
    json.dump(expected_output, f, indent=2)

print("Created example model package.")