
import os
import torch
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
NUM_EPOCHS = 20             
LEARNING_RATE = 1e-3
PRINT_INTERVAL = 50       
CHECKPOINT_INTERVAL = 500    

LAMBDA_CONTENT = 1.0         
LAMBDA_STYLE = 5.0e2       
LAMBDA_TV = 1.0e-6         

CONTENT_ROOT = "/kaggle/input/coco2014/train2014/train2014"     
STYLE_IMAGE_PATH = "/kaggle/input/perceptual-losses/mosaic.jpg"   
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

IMAGE_SIZE = 256

STYLE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

