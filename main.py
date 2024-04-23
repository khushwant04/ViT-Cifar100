import torch
from src.utils.helper import *
from src.models.models import *
from src.datasets.imagenet1k import *
from src.train import *
from tqdm import tqdm

torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader, test_loader = get_cifar100_dataloaders(batch_size=128)
# train_loader, test_loader = get_imagenet_dataloaders(batch_size=64)


# Model parameters
image_size = 224
patch_size = 16
in_channels = 3
num_classes = 100
embed_dim = 768
num_heads = 12
hidden_dim = 3072
num_layers = 12
dropout = 0.1
# Hyperparameters
epochs = 100


# # Initilization
model = VisionTransformer(
    image_size=image_size,
    patch_size=patch_size,
    in_channels=in_channels,
    num_classes=num_classes,
    embed_dim=embed_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=dropout
).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)




trainer = Train(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    accuracy_fn=accuracy_fn,
    device=device,
    lambda_=0.001
)

trainer.train(epochs)
