import os
import torch

from utils.utils_model import CKC
from torch.utils.data import DataLoader
from utils.utils_sr import validate_model
from utils.utils_transform import VideoTransform
from process.busv_dataloader import BUSV_Dataset

if __name__ == '__main__':

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data augmentation and sampling settings
    batch_size = 16
    num_samples = 16
    output_size = 224
    valid_transform = VideoTransform(output_size=output_size, flip_prob=0.0, crop_prob=0.0,
                                     scale=(1.0, 1.0), ratio=(1.0, 1.0))

    # Load the test dataset
    dataset_path = r'E:\MyDataSet\Breast_USV'
    excel_path = r'E:\MyDataSet\Breast_USV\labels.xlsx'
    Data_valid = BUSV_Dataset(os.path.join(dataset_path, "test.txt"), excel_path, base_dir="breast",
                              transform=valid_transform, num_samples=num_samples)
    data_valid = DataLoader(Data_valid, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Initialize the model
    model = CKC(backbone_name='swin_large', num_class=2, channel=1536, num_clinical=8, out_channel=512,
                num_frame=num_samples).cuda()

    # Load the trained model weights
    model_path = 'save_weights/best_model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully and set to evaluation mode.")

    # Validate the model
    avg_loss = validate_model(model, data_valid, device=device)
