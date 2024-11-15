import streamlit as st
from PIL import Image, ImageOps, ExifTags, ImageDraw
import torch
import torchvision.models as models
import torchvision.transforms as transforms

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

# Initialize session state for the image
if "rotated_image" not in st.session_state:
    st.session_state.rotated_image = None
if "cropped_image" not in st.session_state:
    st.session_state.cropped_image = None

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)

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

    # Update the session state with the uploaded image if not rotated yet
    if st.session_state.rotated_image is None:
        st.session_state.rotated_image = image

    # Display the initial uploaded image
    st.image(st.session_state.rotated_image, caption="Uploaded Image", use_column_width=True)

    # Rotate functionality
    st.write("Step 1: Rotate the image if needed.")
    if st.button("Make the Photo Landscape (Swap Width and Height)"):
        st.session_state.rotated_image = st.session_state.rotated_image.transpose(Image.ROTATE_90)
        st.image(st.session_state.rotated_image, caption="Image Rotated to Landscape", use_column_width=True)

    # Manual cropping interface
    st.write("Step 2: Crop the image manually.")
    width, height = st.session_state.rotated_image.size

    # Define sliders for cropping dimensions
    left = st.slider("Select left boundary", 0, width, 0)
    top = st.slider("Select top boundary", 0, height, 0)
    right = st.slider("Select right boundary", left + 1, width, width)
    bottom = st.slider("Select bottom boundary", top + 1, height, height)

    # Create a preview of the crop rectangle on the original image
    preview_image = st.session_state.rotated_image.copy()
    draw = ImageDraw.Draw(preview_image)
    draw.rectangle((left, top, right, bottom), outline="red", width=3)  # Red rectangle for crop preview

    st.image(preview_image, caption="Crop Preview (Red Rectangle)", use_column_width=True)

    # Perform cropping and store in session state
    if st.button("Crop Image"):
        st.session_state.cropped_image = st.session_state.rotated_image.crop((left, top, right, bottom))
        st.success("Image cropped successfully!")

# Display cropped image and classify
if st.session_state.cropped_image:
    st.write("Step 3: Confirm the cropped image.")
    st.image(st.session_state.cropped_image, caption="Cropped Image", use_column_width=True)

    if st.button("Classify"):
        # Preprocess the cropped image
        image = transform(st.session_state.cropped_image).unsqueeze(0)  # Add batch dimension

        # Add a loading spinner
        with st.spinner("Classifying... Please wait."):
            # Run the model
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                denomination = class_names[predicted.item()]

        # Display the prediction
        st.success(f"Predicted Denomination: {denomination}")
