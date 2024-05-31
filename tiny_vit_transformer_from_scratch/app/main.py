import base64
import streamlit as st
from tiny_vit_transformer_from_scratch.model.vit_transformer import VisionTransformer
import torch
from torchvision import transforms
from PIL import Image

def load_from_checkpoints(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = VisionTransformer(**checkpoint['model_args'])
    model.to(device)
    model.eval()

    try:
        model.load_state_dict(checkpoint['model'], strict=True)
    except RuntimeError as e:
        print(f"Failed to load all parameters: {e}")

    optimizer = None
    if 'optimizer_state_dict' in checkpoint:
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model

def add_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    with open("tiny_vit_transformer_from_scratch/app/style/style.css", "r") as css_file:
        css_content = css_file.read()
        css_content = css_content.replace("{encoded_image}", encoded_image)
    st.markdown(
        f"<style>{css_content}</style>",
        unsafe_allow_html=True
    )

def main():
    add_background_image("tiny_vit_transformer_from_scratch/app/src/image.png")
    
    st.markdown('<div class="title-container"><h1 class="title">🖼️ Tiny-ViT Image Classification App</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-container"><p class="instructions">Upload an image, and the Tiny-ViT model will classify it.</p><p class="classes">Current classes the model can detect are: Tomato Bacterial spot, Tomato Early blight, Tomato Late blight, Tomato Leaf Mold</p></div>', unsafe_allow_html=True)
    
    st.sidebar.markdown('<div class="sidebar-title">Tiny-Vit</div>', unsafe_allow_html=True)
    
    checkpoint_path = "C:\\Users\\Omar\\Desktop\\Week-end-projects\\Tiny-ViT-Transformer-from-scratch\\tiny_vit_transformer_from_scratch\\checkpoints\\vit_chpts.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_from_checkpoints(checkpoint_path, device)

    classes = [
        "Tomato Bacterial spot", "Tomato Early blight", "Tomato Late blight", "Tomato Leaf Mold",
        "Tomato Septoria leaf spot", "Tomato Spider mites Two spotted spider mite", "Tomato Target Spot",
        "Tomato healthy", "Potato Early blight", "Potato Late blight", "Tomato Tomato mosaic virus", "Potato healthy"
    ]

    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        im_size = 256 
        image_transforms = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = image_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_class_idx = torch.max(output, 1)
            predicted_class = classes[predicted_class_idx.item()]

            st.markdown(f'<div class="prediction-result">Predicted Class: {predicted_class}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="upload-container"><p class="instructions">Please upload an image to classify.</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
