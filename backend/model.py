import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
import io

class ASLDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        print(f"Dataset initialized with {len(self.data)} samples")
        print(f"Number of features: {len(self.data.columns)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            label = torch.tensor(self.data.iloc[idx, 0], dtype=torch.long)
            image = self.data.iloc[idx, 1:].values.astype(np.uint8).reshape(28, 28)
            image = Image.fromarray(image)
            
            if self.transform:
                image = self.transform(image)
            
            # Ensure image is the correct shape (1, 28, 28)
            if image.shape != (1, 28, 28):
                image = image.reshape(1, 28, 28)
                
            return image, label
        except Exception as e:
            print(f"Error processing index {idx}: {str(e)}")
            raise e

class ASLNet(nn.Module):
    def __init__(self):
        super(ASLNet, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, 3)  # Input: 28x28 -> Output: 26x26
        self.conv2 = nn.Conv2d(32, 64, 3)  # After pool: 13x13 -> Output: 11x11
        self.conv3 = nn.Conv2d(64, 64, 3)  # After pool: 5x5 -> Output: 3x3
        self.pool = nn.MaxPool2d(2, 2)
        # Calculate the correct input size for the first fully connected layer
        self.fc1 = nn.Linear(64 * 1 * 1, 128)  # After final pool: 1x1
        self.fc2 = nn.Linear(128, 24)  # 24 classes (excluding J and Z)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Add print statements to debug shapes
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 13x13
        x = self.pool(F.relu(self.conv2(x)))  # 13x13 -> 5x5
        x = self.pool(F.relu(self.conv3(x)))  # 5x5 -> 1x1
        x = x.view(-1, 64 * 1 * 1)  # Flatten the output
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    best_loss = float('inf')
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Add shape debugging
            if i == 0:  # Print shapes for first batch only
                print(f"Input batch shape: {images.shape}")
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Add shape debugging
            if i == 0:  # Print shapes for first batch only
                print(f"Output batch shape: {outputs.shape}")
                print(f"Labels shape: {labels.shape}")
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                accuracy = 100 * correct / total
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}, Accuracy: {accuracy:.2f}%')
                running_loss = 0.0
                correct = 0
                total = 0
        
        # Save best model
        if running_loss < best_loss:
            best_loss = running_loss
            torch.save(model.state_dict(), 'best_asl_model.pth')

def process_image(image_bytes):
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Define the transformation pipeline to match training
        transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Resize to model's expected input size
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Single channel normalization
        ])
        
        # Apply transformations
        image_tensor = transform(image)
        
        # Ensure correct shape (1, 28, 28)
        if image_tensor.shape != (1, 28, 28):
            image_tensor = image_tensor.reshape(1, 28, 28)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        print(f"Processed image tensor shape: {image_tensor.shape}")  # Debug print
        return image_tensor
    except Exception as e:
        print(f"Error processing image: {str(e)}")  # Debug print
        raise Exception(f"Failed to process image: {str(e)}")

def predict_letter(model, image_tensor, device):
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            # Map the predicted index to the corresponding letter (excluding J and Z)
            letters = 'ABCDEFGHIKLMNOPQRSTUVWXY'  # Note: J and Z removed
            predicted_idx = predicted.item()
            print(f"Raw prediction index: {predicted_idx}")  # Debug print
            return letters[predicted_idx]
    except Exception as e:
        print(f"Error during prediction: {str(e)}")  # Debug print
        raise Exception(f"Failed to make prediction: {str(e)}")

# Training setup code (to be run separately)
if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transforms with explicit grayscale conversion
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure single channel
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load the dataset with debug info
    csv_path = '/Users/armankhan/code/ASL_Recognition_Model/data/sign_mnist_train/sign_mnist_train.csv'
    print(f"Loading dataset from: {csv_path}")
    
    train_dataset = ASLDataset(csv_path, transform=transform)
    
    # Check the first few samples
    print("\nChecking first sample:")
    first_image, first_label = train_dataset[0]
    print(f"First image shape: {first_image.shape}")
    print(f"First label: {first_label}")
    
    # Create train/val split
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    print(f"\nTraining set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    batch_size = 32
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    # Debug first batch
    print("\nChecking first batch:")
    first_batch = next(iter(train_loader))
    first_batch_images, first_batch_labels = first_batch
    print(f"Batch images shape: {first_batch_images.shape}")
    print(f"Batch labels shape: {first_batch_labels.shape}")
    
    model = ASLNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, train_loader, criterion, optimizer, device)
    torch.save(model.state_dict(), 'asl_model.pth')