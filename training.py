import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch

from data.datasets import ContentDataset
from data.augmentation import get_augmentation_transforms
from models.model import TransformerNet
from losses.losses import (
    VGG16LossNetwork,
    compute_content_loss,
    compute_style_loss,
    total_variation_loss
)
import config


def denormalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(img_tensor, 0, 1)

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def visualize_batch(content_images, stylized_images, iteration):
    content_grid = make_grid(content_images, nrow=4)
    stylized_grid = make_grid(stylized_images, nrow=4)

    content_grid = denormalize(content_grid)
    stylized_grid = denormalize(stylized_grid)
    
    plt.figure(figsize=(12,6))
    
    plt.subplot(1,2,1)
    plt.title("Content Images")
    plt.imshow(content_grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    
    plt.subplot(1,2,2)
    plt.title(f"Stylized Images at Iteration {iteration}")
    plt.imshow(stylized_grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    
    save_path = f"vis_{iteration}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")


def load_style_image(style_image_path, transform, device):
    """Load the style image, apply transformation, and return a batch tensor."""
    style_image = Image.open(style_image_path).convert("RGB")
    style_image = transform(style_image).unsqueeze(0)  # Add batch dimension.
    return style_image.to(device)


def main():
    device = config.DEVICE


    style_img = load_style_image(config.STYLE_IMAGE_PATH, config.STYLE_TRANSFORM, device)
    vgg_loss_network = VGG16LossNetwork().to(device)
    vgg_loss_network.eval()

    with torch.no_grad():
        style_features = vgg_loss_network(style_img)

    augmentation = get_augmentation_transforms(config.IMAGE_SIZE)
    content_transform = transforms.Compose([
        augmentation,
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])
    content_dataset = ContentDataset(root_dir=config.CONTENT_ROOT,
                                     image_size=config.IMAGE_SIZE,
                                     transform=content_transform)
    dataloader = DataLoader(content_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=True,
                            num_workers=4)


    transformer_net = TransformerNet().to(device)
    optimizer = optim.Adam(transformer_net.parameters(), lr=config.LEARNING_RATE)

    global_step = 0  
    print("Starting training...")

    for epoch in range(config.NUM_EPOCHS):
        for batch_idx, content_images in enumerate(dataloader):
            transformer_net.train()
            optimizer.zero_grad()

            content_images = content_images.to(device)

            generated_images = transformer_net(content_images)

            vgg_gen_features = vgg_loss_network(generated_images)
            vgg_content_features = vgg_loss_network(content_images)

            content_loss = compute_content_loss(vgg_gen_features['relu2_2'],
                                                vgg_content_features['relu2_2'])

            style_loss = compute_style_loss(vgg_gen_features, style_features)

            tv_loss = total_variation_loss(generated_images)

            total_loss = (config.LAMBDA_CONTENT * content_loss +
                          config.LAMBDA_STYLE * style_loss +
                          config.LAMBDA_TV * tv_loss)

            total_loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % config.PRINT_INTERVAL == 0:
                print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Step [{global_step}], "
                      f"Total Loss: {total_loss.item():.4f}, "
                      f"Content Loss: {content_loss.item():.4f}, "
                      f"Style Loss: {style_loss.item():.4f}, "
                      f"TV Loss: {tv_loss.item():.4f}")
                
                transformer_net.eval()
                with torch.no_grad():
                    generated_images = transformer_net(content_images)

                    generated_images = (generated_images + 1) / 2.0

                    content_disp = (content_images + 1) / 2.0
                transformer_net.train()
                
                visualize_batch(content_disp.cpu(), generated_images.cpu(), global_step)

            if global_step % config.CHECKPOINT_INTERVAL == 0:
                ckpt_path = os.path.join(config.CHECKPOINT_DIR,
                                         f"transformer_epoch{epoch+1}_step{global_step}.pth")
                torch.save(transformer_net.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

    print("Training complete!")

if __name__ == '__main__':
    main()

