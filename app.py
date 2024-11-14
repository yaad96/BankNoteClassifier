import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from streamlit_cropper import st_cropper

# Define the InceptionV3 model architecture
model = models.inception_v3(pretrained=False, aux_logits=False)  # InceptionV3 requires aux_logits=False for inference
num_classes = 9  # Update to match the saved model's output classes
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the model weights
model.load_state_dict(torch.load("inception_v3_model_trained.pth", map_location=torch.device("cpu")))
model.eval()

# Define image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize to match InceptionV3 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class names for each denomination
class_names = {
    0: '1 Taka', 1: '10 Taka', 2: '100 Taka', 3: '1000 Taka', 4: '2 Taka', 5: '20 Taka', 6: '5 Taka', 7: '50 Taka', 8: '500 Taka'
}

# Streamlit UI setup
st.title("Welcome to Banknote Denomination Classifier")
st.write("Upload an image of a banknote to identify its denomination.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.write("Step 1: Crop the uploaded image to focus on the banknote.")
    cropped_image = st_cropper(image, aspect_ratio=(1, 1))

    # Display the cropped image for user confirmation
    st.write("Step 2: Confirm the cropped image.")
    st.image(cropped_image, caption="Cropped Image", use_column_width=True)

    # Add a button for classification
    if st.button("Classify"):
        # Preprocess the cropped image
        image = transform(cropped_image).unsqueeze(0)  # Add batch dimension

        # Add a loading spinner
        with st.spinner("Classifying... Please wait."):
            # Run the model
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                denomination = class_names[predicted.item()]

        # Display the prediction
        st.success(f"Predicted Denomination: {denomination}")
