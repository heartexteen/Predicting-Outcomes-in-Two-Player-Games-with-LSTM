import torch
import torch.nn
import torch.optim
from vit_pytorch import ViT
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from health_tracker import get_health_amounts
from config import *
from torch_dataset import SFDataset

device = "cuda"

epochs = 100

frame_stack = 16

model = ViT(
    image_size = MODEL_IMG_SIZE[0],
    channels=3 * frame_stack,
    patch_size = 16,
    num_classes = 2,
    dim = 512,
    depth = 1,
    heads = 4,
    mlp_dim = 512
).to(device)

lr = 0.001

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_data = SFDataset(frame_stack=frame_stack)
train_loader = DataLoader(dataset=train_data, batch_size=1024, shuffle=True)

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

    torch.save(model.state_dict(), './trained-vit.pt')
