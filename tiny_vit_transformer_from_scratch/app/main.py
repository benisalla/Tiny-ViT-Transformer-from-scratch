import streamlit as st
from tiny_vit_transformer_from_scratch.model.vit_transformer import VisionTransformer
import torch
from torchvision import transforms
from PIL import Image

def load_from_checkpoints(checkpoint_path, device):
    """
    Loads the Tiny-ViT model checkpoint, maps it to the appropriate device, and initializes the optimizer if present.
    """
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

def add_background_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    add_background_image()
    
    st.title("🖼️ Tiny-ViT Image Classification App")
    st.write("Upload an image, and the Tiny-ViT model will classify it.")
    st.sidebar.title("Settings")
    
    # Path and device setup
    checkpoint_path = "C:\\Users\\Omar\\Desktop\\Week-end-projects\\Tiny-ViT-Transformer-from-scratch\\tiny_vit_transformer_from_scratch\\checkpoints\\vit_chpts.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_from_checkpoints(checkpoint_path, device)
    
    # Define classes
    classes = [
        "Tomato Bacterial spot", "Tomato Early blight", "Tomato Late blight", "Tomato Leaf Mold",
        "Tomato Septoria leaf spot", "Tomato Spider mites Two spotted spider mite", "Tomato Target Spot",
        "Tomato healthy", "Potato Early blight", "Potato Late blight", "Tomato Tomato mosaic virus", "Potato healthy"
    ]
    
    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Image preprocessing
        im_size = 256 
        image_transforms = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = image_transforms(image).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_class_idx = torch.max(output, 1)
            predicted_class = classes[predicted_class_idx.item()]

        st.success(f"**Predicted Class:** {predicted_class}")
    else:
        st.info("Please upload an image to classify.")

if __name__ == "__main__":
    main()