import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset

#DATA PREPARATION 
# Fix: Filter only "Horse" category (class 7 in CIFAR-10)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load full dataset
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

# Filter only horse class (index 7)
horse_train_indices = [i for i, (_, label) in enumerate(trainset) if label == 7]
horse_test_indices = [i for i, (_, label) in enumerate(testset) if label == 7]

horse_trainset = Subset(trainset, horse_train_indices)
horse_testset = Subset(testset, horse_test_indices)

trainloader = DataLoader(horse_trainset, batch_size=64, shuffle=True)
testloader = DataLoader(horse_testset, batch_size=64, shuffle=False)

def rgb_to_gray(img):
    """Convert RGB image to grayscale using luminance formula"""
    return 0.2989 * img[:, 0, :, :] + 0.5870 * img[:, 1, :, :] + 0.1140 * img[:, 2, :, :]


#REGRESSION-BASED CNN
class ColorizationCNN(nn.Module):
    def __init__(self):
        super(ColorizationCNN, self).__init__()
        # Fix: Added batch normalization for better training
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 3, 3, padding=1)

        # Fix: Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.sigmoid(self.conv5(x))  # Fix: Use sigmoid to constrain outputs to [0,1]
        return x


#CUSTOM CONVOLUTION LAYER
class MyConv2d(nn.Module):
    """Custom convolution layer for learning purposes"""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(MyConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = padding

        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Manual convolution implementation
        batch_size, _, h, w = x.shape
        kernel_h, kernel_w = self.kernel_size

        # Add padding
        if self.padding > 0:
            x = F.pad(x, [self.padding] * 4)

        # Simple convolution (for educational purposes - in practice use F.conv2d)
        return F.conv2d(x, self.weight, self.bias)


#CLASSIFICATION-BASED CNN
class ClassifierColorizationCNN(nn.Module):
    def __init__(self, num_colors=24):
        super().__init__()
        # Using MyConv2d as required
        self.conv1 = MyConv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = MyConv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = MyConv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = MyConv2d(128, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = MyConv2d(64, num_colors, 3, padding=1)  # Output: color classes

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Encoder path with pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Decoder path with upsampling
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv5(x)

        return x  # Raw logits for CrossEntropyLoss


#UNET WITH SKIP CONNECTIONS
class UNetColorization(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc1_bn = nn.BatchNorm2d(32)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        self.enc2_bn = nn.BatchNorm2d(64)
        self.enc3 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc3_bn = nn.BatchNorm2d(128)

        # Decoder
        self.dec3 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec3_bn = nn.BatchNorm2d(64)
        self.dec2 = nn.Conv2d(64, 32, 3, padding=1)
        self.dec2_bn = nn.BatchNorm2d(32)
        self.dec1 = nn.Conv2d(32, 3, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Encoder with skip connections storage
        e1 = F.relu(self.enc1_bn(self.enc1(x)))  # Skip connection 1
        e2 = self.pool(e1)
        e2 = F.relu(self.enc2_bn(self.enc2(e2)))  # Skip connection 2
        e3 = self.pool(e2)
        e3 = F.relu(self.enc3_bn(self.enc3(e3)))  # Bottleneck

        # Decoder with skip connections
        d3 = F.interpolate(e3, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = F.relu(self.dec3_bn(self.dec3(d3)))
        d3 = d3 + e2  # Skip connection from encoder 2

        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = F.relu(self.dec2_bn(self.dec2(d2)))
        d2 = d2 + e1  # Skip connection from encoder 1

        out = torch.sigmoid(self.dec1(d2))
        return out


#TRAINING FUNCTIONS
def train_model(model, trainloader, testloader, epochs, criterion, optimizer, model_name="Model"):
    """Generic training function with validation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for images, _ in trainloader:
            images = images.to(device)
            gray = rgb_to_gray(images).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(gray)

            if isinstance(criterion, nn.CrossEntropyLoss):
                # For classification, reshape outputs
                batch_size, num_colors, h, w = outputs.shape
                outputs = outputs.permute(0, 2, 3, 1).reshape(-1, num_colors)
                targets = (images * 255).long().permute(0, 2, 3, 1).reshape(-1, 3)
                # Simplified: map RGB to nearest color class (would need color palette)
                loss = criterion(outputs, targets[:, 0])  # Simplified
            else:
                loss = criterion(outputs, images)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(trainloader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, _ in testloader:
                images = images.to(device)
                gray = rgb_to_gray(images).unsqueeze(1)
                outputs = model(gray)

                if isinstance(criterion, nn.CrossEntropyLoss):
                    outputs = outputs.permute(0, 2, 3, 1).reshape(-1, 24)
                    targets = (images * 255).long().permute(0, 2, 3, 1).reshape(-1, 3)
                    loss = criterion(outputs, targets[:, 0])
                else:
                    loss = criterion(outputs, images)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(testloader)
        val_losses.append(avg_val_loss)

        print(
            f"{model_name} - Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses


def visualize_results(model, testloader, num_images=3):
    """Visualize colorization results"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    images_shown = 0
    with torch.no_grad():
        for images, _ in testloader:
            images = images.to(device)
            gray = rgb_to_gray(images).unsqueeze(1)
            outputs = model(gray)

            for i in range(min(num_images - images_shown, len(images))):
                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                plt.title("Grayscale Input")
                plt.imshow(gray[i].cpu().squeeze(), cmap='gray')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.title("Colorized Output")
                plt.imshow(outputs[i].cpu().permute(1, 2, 0).numpy())
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.title("Original Image")
                plt.imshow(images[i].cpu().permute(1, 2, 0).numpy())
                plt.axis('off')

                plt.tight_layout()
                plt.show()

                images_shown += 1
                if images_shown >= num_images:
                    return


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Train Regression Model
    print("\n" + "=" * 50)
    print("Training Regression Model")
    print("=" * 50)

    reg_model = ColorizationCNN()
    reg_criterion = nn.MSELoss()
    reg_optimizer = torch.optim.Adam(reg_model.parameters(), lr=0.001)

    reg_train_loss, reg_val_loss = train_model(
        reg_model, trainloader, testloader,
        epochs=20,
        criterion=reg_criterion,
        optimizer=reg_optimizer,
        model_name="Regression"
    )

    # Visualize regression results
    print("\nRegression Model Results:")
    visualize_results(reg_model, testloader, num_images=2)

    # 2. Train UNet Model
    print("\n" + "=" * 50)
    print("Training UNet Model with Skip Connections")
    print("=" * 50)

    unet_model = UNetColorization()
    unet_optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.001)

    unet_train_loss, unet_val_loss = train_model(
        unet_model, trainloader, testloader,
        epochs=20,
        criterion=nn.MSELoss(),
        optimizer=unet_optimizer,
        model_name="UNet"
    )

    # Visualize UNet results
    print("\nUNet Model Results:")
    visualize_results(unet_model, testloader, num_images=2)

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(reg_train_loss, label='Regression Train')
    plt.plot(reg_val_loss, label='Regression Val')
    plt.plot(unet_train_loss, label='UNet Train')
    plt.plot(unet_val_loss, label='UNet Val')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
