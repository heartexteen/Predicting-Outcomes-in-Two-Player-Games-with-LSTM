import torch
import torch.nn
import torch.optim
import torchvision.models
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from health_tracker import get_health_amounts
from config import *
from torch_dataset import SFDataset

device = "cuda"

epochs = 1000

frame_stack = 1

model = torchvision.models.convnext_small(num_classes=2).to(device)

lr = 0.001

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_data = SFDataset(frame_stack=frame_stack)
train_loader = DataLoader(dataset=train_data, batch_size=512, shuffle=True)

for epoch in range(epochs):
    epoch_loss = 0

    for idx, (data, label) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss / len(train_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f}"
    )

    torch.save(model.state_dict(), './trained-conv.pt')
