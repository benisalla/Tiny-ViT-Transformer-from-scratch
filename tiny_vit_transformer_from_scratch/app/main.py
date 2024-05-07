import streamlit as st
from tiny_vit_transformer_from_scratch.core.config import VitConfig
from tiny_vit_transformer_from_scratch.model.vit_transformer import VisionTransformer
import torch
from torchvision import transforms
from PIL import Image


def load_from_checkpoints(checkpoint_path):
    """
    Loads the Tiny-ViT model checkpoint and handles CPU mapping.
    """
    device = torch.device('cpu') 
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = VisionTransformer(**checkpoint['model_args'])
    model.load_state_dict(checkpoint['model'])

    optimizer = None
    if 'optimizer_state_dict' in checkpoint:
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, checkpoint

def main():
    checkpoint_path = "C:\\Users\\Omar\\Desktop\\Week-end-projects\\Tiny-ViT-Transformer-from-scratch\\tiny_vit_transformer_from_scratch\\checkpoints\\vit_chpts.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_from_checkpoints(checkpoint_path)
    model = model.to(device)
    model.eval()

    st.title("Tiny-ViT Image Classification App")
    st.write("Upload an image, and the Tiny-ViT model will classify it.")

    # Allow selection of image size through a slider
    im_size = st.slider("Image Size", min_value=64, max_value=512, value=224, step=32)

    image_transforms = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Upload image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        input_tensor = image_transforms(image).unsqueeze(0).to(device)

        # Predict the class using the model
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_class = torch.max(output, 1)

        # Display the predicted class
        st.write(f"Predicted Class: {predicted_class.item()}")

if __name__ == "__main__":
    main()
