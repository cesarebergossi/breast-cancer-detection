# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import config

# ------------------------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------------------------

# Model for segmentation
segmentation_model_path = os.path.join(config.PTH_SAVED_MODELS,'2024-05-24_15_35_01_model.pth')
segmentation_model = torch.load(segmentation_model_path, map_location=torch.device('cpu'))
# segmentation_model.eval()

# Model for classification
classification_model_path = os.path.join(config.PTH_SAVED_MODELS,'2024-05-27_22_27_21_model.pth')
prediction_model = torch.load(classification_model_path, map_location=torch.device('cpu'))
# prediction_model.eval()

# ------------------------------------------------------------------------------------------------
# Transformations
# ------------------------------------------------------------------------------------------------

# Transformation for classification
classification_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Transformation for segmentation
segmentation_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),  # Convert to grayscale
    transforms.ToTensor()
])

upscale_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),  # Convert to grayscale
    transforms.ToTensor()
])

# ------------------------------------------------------------------------------------------------
# Inference functions
# ------------------------------------------------------------------------------------------------

# Function to perform segmentation
def segment_image(model,image):
    model.eval()
    image_transformed = segmentation_transform(image).unsqueeze(0) # make sure 2d tensor
    with torch.no_grad():
        logits = model(image_transformed)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Convert probabilities to binary predictions (0 or 1)
        mask = (probs >= 0.5).squeeze().cpu().numpy().astype(int)
    return mask


# Function to perform prediction
def predict_image(model, image):
    model.eval()
    image_transformed = classification_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_transformed)
        probs = F.softmax(output, dim=1)
        class_idx = torch.argmax(probs, dim=1).item()
        return class_idx, probs[0][class_idx].item()

# ------------------------------------------------------------------------------------------------
# Streamlit 
# ------------------------------------------------------------------------------------------------

# Streamlit interface
st.title("Breast Cancer Detection and Segmentation")
st.write("Upload a scan to get the cancer type and corresponding segmentation.")
uploaded_file = st.file_uploader("Choose a scan...", type="png")

# testing
# uploaded_file = '/Users/eliaparolari/Desktop/23-24/Deep_learning/g_drive_proj/data/raw/malignant/malignant (1).png'


if uploaded_file is not None:
    try:
        # Load the image
        image = Image.open(uploaded_file)
        
        # Perform prediction
        class_idx, confidence = predict_image(prediction_model, image)
        
        # class name
        class_names = ['benign', 'malignant', 'normal']
        cancer_type = class_names[class_idx]
        
        if cancer_type != 'normal':
            # Perform segmentation
            segmentation_mask = segment_image(segmentation_model,image) 
        else:
            segmentation_mask = np.zeros((64, 64)) # black mask

        # segmentation mask resize
        segmentation_mask_res = cv2.resize(segmentation_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        image_bew = np.array(image.convert('L'))
        image_bew = cv2.resize(image_bew, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Convert the grayscale image to a BGR image
        image_bew_bgr = cv2.cvtColor(image_bew, cv2.COLOR_GRAY2BGR)
        
        # Create a blue version of the segmentation mask
        red_mask = np.zeros_like(image_bew_bgr)
        red_mask[:, :, 2] = segmentation_mask_res * 255  # Set the blue channel
        
        # Set the opacity level (0 to 1)
        opacity = 0.2
        
        # Combine the images using addWeighted
        overlay = cv2.addWeighted(red_mask, opacity, image_bew_bgr, 1 - opacity, 0)
        
        # Convert back to an image for visualization (optional)
        overlay_image = Image.fromarray(overlay)

        # Display the prediction
        if cancer_type != 'normal':
            st.write(f"Predicted Class: {cancer_type} with confidence {confidence:.2f}")
        else:
            st.write(f"No cancer detected")
        

        st.image(overlay_image, caption='Scanned image with segmentation', use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# To view the app on a browser, run the following on the terminal:
# streamlit run app.py --server.enableXsrfProtection false
# The last part after -- avoids some problems