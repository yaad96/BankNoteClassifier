import streamlit as st
from PIL import Image, ExifTags
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from streamlit_cropper import st_cropper

st.set_page_config(layout="wide")


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
st.title("Bangladeshi Banknote Denomination Classifier")
st.write("Upload an image of a banknote to identify its denomination.")



# Initialize session state for the image
if "rotated_image" not in st.session_state:
    st.session_state.rotated_image = None
if "cropped_image" not in st.session_state:
    st.session_state.cropped_image = None

# File uploader
uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)

    # Ensure the image is in RGB format
    if image.mode != "RGB":
        image = image.convert("RGB")

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

    # Get the height and width of the image
    original_width, original_height = image.size

    # Resize the image to width=400 while maintaining aspect ratio
    new_width = 400
    aspect_ratio = original_height / original_width
    new_height = int(new_width * aspect_ratio)
    image = image.resize((new_width, new_height))

    # Update the session state with the resized image
    st.session_state.rotated_image = image

    # Display the resized image
    st.image(st.session_state.rotated_image, caption="Resized Image", use_column_width=True)

    # Perform cropping using Streamlit-Cropper
    st.write("Step 1: Crop the image using the tool below.")
    cropped_image = st_cropper(st.session_state.rotated_image, realtime_update=True, box_color="red", aspect_ratio=None)

    if st.button("Confirm Crop", key="crop_button"):
        st.session_state.cropped_image = cropped_image
        st.success("Image cropped successfully!")

# Rotate cropped image if needed
if st.session_state.cropped_image:
    st.write("Step 2: Rotate the cropped image if needed to make the note horizontal.")
    st.image(st.session_state.cropped_image, caption="Cropped Image", use_column_width=True)

    if st.button("Make the note horizontal", key="rotate_button"):
        st.session_state.cropped_image = st.session_state.cropped_image.transpose(Image.ROTATE_90)
        st.image(st.session_state.cropped_image, caption="Note Rotated", use_column_width=True)

# Classify the image
if st.session_state.cropped_image:
    st.write("Step 3: Classify the image.")

    if st.button("Classify", key="classify_button"):
        # Preprocess the cropped (and potentially rotated) image for classification
        final_image = transform(st.session_state.cropped_image).unsqueeze(0)  # Add batch dimension

        # Add a loading spinner
        with st.spinner("Classifying... Please wait."):
            # Run the model
            with torch.no_grad():
                output = model(final_image)
                _, predicted = torch.max(output, 1)
                denomination = class_names[predicted.item()]

        # Display the prediction
        st.success(f"Predicted Denomination: {denomination}")

        # Add a "Classify another note" button with a unique key
        if st.button("Classify Another Note", key="classify_another_button"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]

            st.warning("Please upload a new image to start fresh.")
