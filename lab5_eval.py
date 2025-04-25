import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from PIL import Image


val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


data_dir = '/home/nastia/agh/biometrics/fvc2000'
data_transforms = {
    'train': val_transform,
    'val': val_transform
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
class_names = image_datasets['train'].classes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load('lab5_fvc2000_model.pth', map_location=device))
model = model.to(device)
model.eval()

test_dir = '/home/nastia/agh/biometrics/fvc2000_final_test'
image_files = [f for f in os.listdir(test_dir) if f.endswith('.bmp')]
image_files.sort()

for fname in image_files:
    fpath = os.path.join(test_dir, fname)
    image = Image.open(fpath).convert('RGB')
    input_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        predicted_class = class_names[pred.item()]

    print(f"{fname}: Predicted class → {predicted_class}")


