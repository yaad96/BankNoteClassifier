import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms



# Define the model architecture
model = models.vgg16(pretrained=False)
num_classes = 9  # Update to match the saved model's output classes
model.classifier[6] = torch.nn.Linear(4096, num_classes)

# Load the model weights
model.load_state_dict(torch.load("vgg16_model_trained.pth", map_location=torch.device("cpu")))
model.eval()

# Define image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class names for each denomination
# Ensure you have 9 classes here to match the model's output
class_names = {
    0: '1 Taka', 1: '10 Taka', 2: '100 Taka', 3: '1000 Taka', 4: '2 Taka', 5: '20 Taka', 6: '5 Taka', 7: '50 Taka', 8: '500 Taka'
}

# Streamlit UI setup
st.title("Welcadasdome to Banknote Denomination Classifier")
st.write("Upload an image of a banknote to identify its denomination.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Add a loading spinner
    with st.spinner("Classifying... Please wait."):
        # Run the model
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            denomination = class_names[predicted.item()]
            print("Predicted: ", predicted)
            print("denomination: ", denomination)

    # Display the prediction
    st.success(f"Predicted Denomination: {denomination}")
