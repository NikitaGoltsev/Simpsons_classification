from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np

import torch
import glob
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, class_num, transform=None):
        self.path      = path
        self.class_num = class_num
        self.transform = transform

        self.file_paths = []
        self.tags = []

        for class_name in os.listdir(self.path):
            class_dir = os.path.join(self.path, class_name)
            if not os.path.isdir(class_dir):
                continue
            tag = class_num.get(class_name)
            if tag is None:
                continue
            class_file_paths = glob.glob(os.path.join(class_dir, "*.jpg"))
            self.file_paths.extend(class_file_paths)
            self.tags.extend([tag] * len(class_file_paths))

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        tag = self.tags[idx]
        with open(img_path, "rb") as image_file:
            img = Image.open(image_file).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(tag, dtype=torch.long)

path_test = "simpson/simpsons_test"
path_train = "simpson/simpsons_train"

transform_for_test = transforms.Compose([
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def print_graphic(graph):
    fig = plt.figure(figsize=(8,5))
    idx = np.arange(0, len(graph))
    plt.plot(idx, graph)
    plt.xlabel('Images')
    plt.ylabel('Accurasy')
    plt.show()


def test_image(model, img_path, class_num):
    with open(img_path, "rb") as f:
        img = Image.open(f).convert("RGB")
    img_tensor = transform_for_test(img).unsqueeze(0).to(device)

    output = model(img_tensor)
    probs = torch.softmax(output, dim=1)

    pred_label = torch.argmax(probs, dim=1).item()
    class_name = list(class_num.keys())[list(class_num.values()).index(pred_label)]

    return pred_label, class_name, probs.squeeze().detach().cpu().numpy()


def main_train():
    batch_size = 32
    class_num = {}

    i = 0
    for classes_name in os.listdir(path_train):
        class_num[classes_name] = i
        i += 1

    dataset = CustomImageDataset(path_train, class_num, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True).to(device)

    num_classes = len(class_num)

    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, num_classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    num_epochs = 3

    for epoch in range(num_epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            print(f"epoch = {epoch+1}  loss = {loss:.4f} ")

    torch.save(model.state_dict(), 'modelMNnew2.pth')


def main_test():
    graphic = []
    class_num = {}
    i = 0
    for classes_name in os.listdir(path_train):
        class_num[classes_name] = i
        i += 1

    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False).to(device)
    model.eval()

    num_classes = 42

    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, num_classes)
    ).to(device)

    right = 0
    wrong = 0

    model.load_state_dict(torch.load('modelMNnew2.pth'))

    for i in os.listdir(path_test):
        pred_label, class_name, probs = test_image(model, os.path.join(path_test, i), class_num)
        if (i.find(class_name) != -1):
            # print(f"{right}/{wrong+right} Character: {i}, answer: {class_name}")
            right += 1
        else:
            print(f"{wrong}/{wrong+right} Character: {i}, answer: {class_name}")
            wrong += 1
        graphic.append(right/(right+wrong))

    print(f"right {right}, wrong {wrong}")
    print(f"accurasy: {(right / (right + wrong) * 100):.2f}%")
    print_graphic(graphic)


if __name__ == '__main__':
    main_test()