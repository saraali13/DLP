"""
DCGAN for Handwritten Digit Generation on MNIST Dataset
Assignment: Handwritten Digit Generation with DCGAN
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
latent_dim = 100          # Size of noise vector (z)
batch_size = 64           # Batch size for training (reduced for CPU performance)
learning_rate = 0.0002    # Learning rate (following DCGAN paper)
num_epochs = 20           # Number of training epochs (reduced from 50 for CPU performance)
image_size = 32           # Reduced from 64x64 for faster CPU training
channels_img = 1          # Grayscale images
channels_noise = latent_dim

# Create directories for saving outputs
os.makedirs("generated_images", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Data preprocessing and loading
transform = transforms.Compose([
    transforms.Resize(image_size),           # Resize 28x28 to 64x64
    transforms.ToTensor(),                   # Convert to tensor [0,1]
    transforms.Normalize([0.5], [0.5])       # Normalize to [-1, 1] (better for GANs)
])

# Download and load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
)

print(f"Dataset loaded: {len(train_dataset)} training samples")

# ==================== Generator Network ====================
class Generator(nn.Module):
    """
    DCGAN Generator - Takes random noise and generates fake images (32x32)
    Architecture following DCGAN paper guidelines:
    - No pooling layers (use strided convolutions for upsampling)
    - BatchNorm after every layer except output
    - ReLU activation for all layers except output (Tanh)
    """
    def __init__(self, latent_dim, channels_img, feature_g=64):
        super(Generator, self).__init__()
        
        self.gen = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_g * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            # Output: (feature_g*4) x 4 x 4
            
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            # Output: (feature_g*2) x 8 x 8
            
            nn.ConvTranspose2d(feature_g * 2, feature_g, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            # Output: feature_g x 16 x 16
            
            nn.ConvTranspose2d(feature_g, channels_img, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Output: channels_img x 32 x 32
        )
    
    def forward(self, z):
        # Reshape noise to 4D tensor: (batch, latent_dim, 1, 1)
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.gen(z)


# ==================== Discriminator Network ====================
class Discriminator(nn.Module):
    """
    DCGAN Discriminator - Distinguishes real from fake images (32x32)
    Architecture following DCGAN paper guidelines:
    - No pooling layers (use strided convolutions for downsampling)
    - BatchNorm after every layer except input
    - LeakyReLU activation
    """
    def __init__(self, channels_img, feature_d=64):
        super(Discriminator, self).__init__()
        
        self.disc = nn.Sequential(
            # Input: channels_img x 32 x 32
            nn.Conv2d(channels_img, feature_d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: feature_d x 16 x 16
            
            nn.Conv2d(feature_d, feature_d * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (feature_d*2) x 8 x 8
            
            nn.Conv2d(feature_d * 2, feature_d * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (feature_d*4) x 4 x 4
            
            nn.Conv2d(feature_d * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1 (probability)
        )
    
    def forward(self, x):
        return self.disc(x).view(-1, 1)


# Initialize networks
generator = Generator(latent_dim, channels_img).to(device)
discriminator = Discriminator(channels_img).to(device)

# Initialize weights (following DCGAN paper: mean=0, std=0.02)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator.apply(weights_init)
discriminator.apply(weights_init)

print("="*50)
print("Generator Architecture:")
print(generator)
print("\n" + "="*50)
print("Discriminator Architecture:")
print(discriminator)
print("="*50)

# Loss function and optimizers
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# For tracking losses
G_losses = []
D_losses = []

# Fixed noise for visualizing progress
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

print("\nStarting Training...")
print(f"Total epochs: {num_epochs}")
print(f"Batch size: {batch_size}")
print(f"Batches per epoch: {len(train_loader)}")

# Training loop
for epoch in range(num_epochs):
    epoch_g_loss = 0
    epoch_d_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    
    for batch_idx, (real_images, _) in enumerate(progress_bar):
        batch_size_curr = real_images.size(0)
        real_images = real_images.to(device)
        
        # ==================== Train Discriminator ====================
        # Real images: label = 1 (real)
        real_labels = torch.ones(batch_size_curr, 1).to(device)
        fake_labels = torch.zeros(batch_size_curr, 1).to(device)
        
        # Forward pass real images through discriminator
        real_output = discriminator(real_images)
        d_loss_real = criterion(real_output, real_labels)
        
        # Generate fake images
        noise = torch.randn(batch_size_curr, latent_dim, device=device)
        fake_images = generator(noise)
        fake_output = discriminator(fake_images.detach())  # detach to avoid training generator
        d_loss_fake = criterion(fake_output, fake_labels)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        
        # Backprop for discriminator
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        
        # ==================== Train Generator ====================
        # Generate fake images again (for generator training)
        noise = torch.randn(batch_size_curr, latent_dim, device=device)
        fake_images = generator(noise)
        fake_output = discriminator(fake_images)
        
        # Generator wants discriminator to think these are real (label=1)
        g_loss = criterion(fake_output, real_labels)
        
        # Backprop for generator
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
        
        # Store losses
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'D_loss': f'{d_loss.item():.4f}',
            'G_loss': f'{g_loss.item():.4f}'
        })
    
    # Average losses for the epoch
    avg_g_loss = epoch_g_loss / len(train_loader)
    avg_d_loss = epoch_d_loss / len(train_loader)
    G_losses.append(avg_g_loss)
    D_losses.append(avg_d_loss)
    
    # Generate and save sample images every 2 epochs
    if (epoch + 1) % 2 == 0:
        generator.eval()
        with torch.no_grad():
            fake_samples = generator(fixed_noise)
            save_image(fake_samples, f'generated_images/epoch_{epoch+1}.png', 
                      nrow=8, normalize=True)
        generator.train()
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}] '
              f'D_loss: {avg_d_loss:.4f} '
              f'G_loss: {avg_g_loss:.4f}')
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'G_loss': avg_g_loss,
            'D_loss': avg_d_loss,
        }, f'models/checkpoint_epoch_{epoch+1}.pth')

print("\nTraining Complete!")

# Save final models
torch.save(generator.state_dict(), 'models/generator_final.pth')
torch.save(discriminator.state_dict(), 'models/discriminator_final.pth')

# ==================== Visualization and Evaluation ====================

def plot_training_losses(G_losses, D_losses):
    """Plot the training losses over epochs"""
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss', color='blue')
    plt.plot(D_losses, label='Discriminator Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_losses.png', dpi=150, bbox_inches='tight')
    plt.show()

def generate_and_display_images(generator, num_images=25):
    """Generate and display a grid of fake digits"""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, latent_dim, device=device)
        generated_images = generator(noise)
        # Denormalize from [-1, 1] to [0, 1] for display
        generated_images = (generated_images + 1) / 2
        
        # Create grid
        grid = make_grid(generated_images, nrow=5, padding=2, normalize=False)
        
        # Convert to numpy for display
        grid = grid.cpu().numpy().transpose(1, 2, 0)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid, cmap='gray')
        plt.axis('off')
        plt.title(f'Generated Handwritten Digits ({num_images} samples)', fontsize=14)
        plt.savefig('generated_digits_final.png', dpi=150, bbox_inches='tight')
        plt.show()
    generator.train()

def save_individual_digits(generator, num_images=10):
    """Save individual generated digits for submission"""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, latent_dim, device=device)
        generated_images = generator(noise)
        generated_images = (generated_images + 1) / 2
        
        # Save each image individually
        for i in range(num_images):
            img = generated_images[i].cpu().squeeze()
            plt.figure(figsize=(2, 2))
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.savefig(f'generated_images/digit_{i+1}.png', 
                       dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
    generator.train()
    print(f"Saved {num_images} individual generated digits to 'generated_images/' directory")

# Plot training losses
plot_training_losses(G_losses, D_losses)

# Generate and display final images
print("\nGenerating final sample images...")
generate_and_display_images(generator, num_images=25)

# Save individual digits for deliverables
save_individual_digits(generator, num_images=10)

# ==================== Quality Evaluation Metrics ====================

# Simple CNN classifier for Inception Score computation
class SimpleClassifier(nn.Module):
    """Simple CNN to classify MNIST digits for Inception Score"""
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_classifier_for_metrics(train_loader, device, epochs=5):
    """Train a simple classifier on real MNIST data for evaluation metrics"""
    classifier = SimpleClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    
    print("\nTraining classifier for quality metrics...")
    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"Classifier training - Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    classifier.eval()
    return classifier

def compute_inception_score(generator, classifier, n_samples=1000, batch_size_eval=100):
    """
    Compute Inception Score using a trained classifier
    Higher score indicates better quality and diversity
    """
    generator.eval()
    classifier.eval()
    
    all_predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples // batch_size_eval):
            noise = torch.randn(batch_size_eval, latent_dim, device=device)
            fake_images = generator(noise)
            
            # Resize to 64x64 if needed for classifier
            predictions = classifier(fake_images)
            all_predictions.append(predictions.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    predictions_probs = np.exp(all_predictions) / np.exp(all_predictions).sum(axis=1, keepdims=True)
    
    # Compute Inception Score
    p_y = np.mean(predictions_probs, axis=0)
    inception_score = np.exp(np.mean([np.sum(p * np.log(p / p_y)) for p in predictions_probs]))
    
    generator.train()
    return inception_score

def compute_fid_score(generator, train_loader, device, n_samples=1000):
    """
    Compute simplified Fréchet Inception Distance (FID)
    Lower FID indicates better quality of generated images
    """
    # Feature extractor (using a simple layer from discriminator-like architecture)
    feature_extractor = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
    ).to(device)
    
    generator.eval()
    
    # Collect real image features
    real_features = []
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            features = feature_extractor(images)
            features = features.view(features.size(0), -1)
            real_features.append(features.cpu().numpy())
    
    real_features = np.concatenate(real_features, axis=0)
    
    # Collect fake image features
    fake_features = []
    with torch.no_grad():
        for _ in range((n_samples // batch_size) + 1):
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise)
            features = feature_extractor(fake_images)
            features = features.view(features.size(0), -1)
            fake_features.append(features.cpu().numpy())
    
    fake_features = np.concatenate(fake_features, axis=0)[:n_samples]
    
    # Compute mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    
    sigma_real = np.cov(real_features.T)
    sigma_fake = np.cov(fake_features.T)
    
    # Compute FID
    diff = mu_real - mu_fake
    cov_sqrt = np.linalg.inv(np.linalg.cholesky(sigma_fake + 1e-6))
    fid = np.trace(sigma_real + sigma_fake - 2 * cov_sqrt @ sigma_real @ cov_sqrt.T)
    
    generator.train()
    return np.sqrt(np.maximum(fid, 0))

def compare_real_vs_generated(generator, train_loader, device, num_pairs=5):
    """Generate side-by-side comparison of real vs generated digits"""
    generator.eval()
    
    # Get real images
    real_images, _ = next(iter(train_loader))
    real_images = real_images[:num_pairs].to(device)
    
    # Generate fake images
    with torch.no_grad():
        noise = torch.randn(num_pairs, latent_dim, device=device)
        fake_images = generator(noise)
    
    # Denormalize
    real_images = (real_images + 1) / 2
    fake_images = (fake_images + 1) / 2
    
    # Create comparison figure
    fig, axes = plt.subplots(2, num_pairs, figsize=(15, 4))
    fig.suptitle('Real vs Generated MNIST Digits Comparison', fontsize=16, fontweight='bold')
    
    for i in range(num_pairs):
        # Real images
        axes[0, i].imshow(real_images[i].cpu().squeeze(), cmap='gray')
        axes[0, i].set_title(f'Real Digit {i+1}', fontsize=10)
        axes[0, i].axis('off')
        
        # Generated images
        axes[1, i].imshow(fake_images[i].cpu().squeeze(), cmap='gray')
        axes[1, i].set_title(f'Generated Digit {i+1}', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('real_vs_generated_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    generator.train()
    print(f"Saved comparison of {num_pairs} real vs generated digits")

# Generate a larger batch and compute metrics
print("\n" + "="*50)
print("Computing Quality Metrics...")
print("="*50)

# Skip intensive metrics on CPU for speed - uncomment if you have GPU
if device.type == 'cuda':
    # Train classifier for Inception Score
    classifier = train_classifier_for_metrics(train_loader, device, epochs=5)

    # Compute Inception Score
    print("\nComputing Inception Score...")
    inception_score = compute_inception_score(generator, classifier, n_samples=500)
    print(f"Inception Score: {inception_score:.4f}")
    print("(Higher is better - indicates better quality and diversity)")

    # Compute FID Score
    print("\nComputing Fréchet Inception Distance (FID)...")
    fid_score = compute_fid_score(generator, train_loader, device, n_samples=500)
    print(f"FID Score: {fid_score:.4f}")
    print("(Lower is better - indicates similarity to real distribution)")

    # Real vs Generated Comparison
    print("\nGenerating Real vs Generated comparison...")
    compare_real_vs_generated(generator, train_loader, device, num_pairs=5)
else:
    print("\nSkipping intensive metric computations on CPU for speed.")
    print("To compute metrics, please use a GPU or reduce batch_size further.")

print("\n" + "="*50)
print("\nDeliverables generated:")
print("1. Training code with DCGAN implementation")
print("2. Generated sample images (saved in 'generated_images/' directory)")
print("3. Training loss plot ('training_losses.png')")
print("4. Final grid of generated digits ('generated_digits_final.png')")
print("5. Model checkpoints saved in 'models/' directory")
print("\n10 individual digit images have been saved as 'digit_1.png' through 'digit_10.png'")
