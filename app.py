# Import required libraries
import streamlit as st
from PIL import Image, ImageOps, ExifTags
import io
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
st.title("Welcome to BD Banknote Denomination Classifier")
st.write("Upload an image of a banknote to identify its denomination.")


# Function to resize the image while maintaining aspect ratio
def resize_image(image, base_size=1024):
    max_dim = max(image.size)
    scale_factor = base_size / max_dim
    new_size = tuple([int(dim * scale_factor) for dim in image.size])
    return image.resize(new_size, Image.Resampling.LANCZOS)  # Updated from ANTIALIAS to LANCZOS


# If not classified, show file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "heic", "heif"])

if uploaded_file is not None:
    # Open the image
    try:
        image = Image.open(uploaded_file)
    except IOError:
        st.error("Could not open the uploaded image. Please ensure it is in a supported format.")

    # Convert to JPG/PNG format
    output_format = "PNG"
    buffer = io.BytesIO()
    image = image.convert("RGB")  # Ensure compatibility for all formats
    image.save(buffer, format=output_format)
    buffer.seek(0)
    image = Image.open(buffer)

    # Resize the image to ensure it fits in the cropping window
    image = resize_image(image, base_size=1024)

    # Fix image orientation (EXIF metadata)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = image._getexif()
        if exif is not None and orientation in exif:
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        pass  # If there's an issue with EXIF, proceed without adjustments

    # Display the resized and reoriented image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Cropping functionality
    st.write("Step 1: Crop the image to focus on the banknote.")
    cropped_image = st_cropper(
        image,
        aspect_ratio=(1, 1),
        box_color="#00008B",  # Set the box and anchors to darkest blue
        realtime_update=True  # Ensures updates as you move the crop box
    )

    # Display the cropped image
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
