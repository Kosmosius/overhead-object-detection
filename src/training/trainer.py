# src/training/trainer.py

import torch
from torch.utils.data import DataLoader
from transformers import DetrFeatureExtractor
from src.models.foundation_model import DetrObjectDetectionModel
from src.evaluation.metrics.py import evaluate_model
from torchvision.datasets import CocoDetection
from transformers import AdamW
import os

def train(model, dataloader, optimizer, num_epochs=10, device='cuda'):
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0

        for i, (images, targets) in enumerate(dataloader):
            optimizer.zero_grad()

            # Prepare inputs
            pixel_values = feature_extractor(images=list(images), return_tensors="pt").pixel_values.to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            outputs = model.forward(pixel_values=pixel_values)
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                print(f"  Batch {i}, Loss: {loss.item()}")

        print(f"  Epoch Loss: {running_loss / len(dataloader)}")
    
    print("Training complete.")

if __name__ == "__main__":
    # Define constants
    DATA_DIR = "/path/to/coco/dataset"  # Replace with your dataset path
    BATCH_SIZE = 2
    NUM_CLASSES = 91  # For COCO dataset
    NUM_EPOCHS = 5
    LEARNING_RATE = 5e-5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the feature extractor, model, optimizer, and dataset
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = DetrObjectDetectionModel(num_classes=NUM_CLASSES)
    optimizer = AdamW(model.model.parameters(), lr=LEARNING_RATE)

    # Prepare the COCO dataset and dataloader
    train_dataset = CocoDetection(root=os.path.join(DATA_DIR, 'train2017'),
                                  annFile=os.path.join(DATA_DIR, 'annotations/instances_train2017.json'),
                                  transform=lambda x: x)  # Placeholder transform
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    val_dataset = CocoDetection(root=os.path.join(DATA_DIR, 'val2017'),
                                annFile=os.path.join(DATA_DIR, 'annotations/instances_val2017.json'),
                                transform=lambda x: x)  # Placeholder transform
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Train the model
    train(model, train_loader, optimizer, num_epochs=NUM_EPOCHS, device=DEVICE)
    
    # Evaluate the model
    precision, recall = evaluate_model(model, val_loader, device=DEVICE)
    print(f"Validation Precision: {precision}, Recall: {recall}")

    # Save the trained model
    model.save("output/detr_model")
