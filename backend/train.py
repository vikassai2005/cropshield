"""Train MobileNetV3-Small on extracted PlantVillage data.
Run: python backend/train.py --data data/PlantVillage/train --epochs 8
"""
import argparse, json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/PlantVillage/train")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--samples-per-class", type=int, default=200)
args = parser.parse_args()
root = Path(args.data)
weights = MobileNet_V3_Small_Weights.DEFAULT
dataset = datasets.ImageFolder(root, transform=weights.transforms())
# Class-balanced first pass: every disease receives the same training budget.
chosen = []
for class_id in range(len(dataset.classes)):
    chosen.extend([i for i, target in enumerate(dataset.targets) if target == class_id][:args.samples_per_class])
balanced = Subset(dataset, chosen)
train_size = int(len(balanced) * .9)
train_set, valid_set = random_split(balanced, [train_size, len(balanced)-train_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=0)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = mobilenet_v3_small(weights=weights)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(dataset.classes))
for parameter in model.features.parameters():
    parameter.requires_grad = False
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(args.epochs):
    model.train()
    for images, targets in train_loader:
        optimizer.zero_grad(); loss = loss_fn(model(images.to(device)), targets.to(device)); loss.backward(); optimizer.step()
    model.eval(); correct = total = 0
    with torch.inference_mode():
        for images, targets in valid_loader:
            correct += (model(images.to(device)).argmax(1).cpu() == targets).sum().item(); total += len(targets)
    print(f"epoch {epoch+1}/{args.epochs}: validation accuracy {correct/total:.2%}")
artifacts = Path("artifacts"); artifacts.mkdir(exist_ok=True)
torch.save(model.cpu().state_dict(), artifacts / "crop_classifier.pt")
(artifacts / "labels.json").write_text(json.dumps(dataset.classes))
