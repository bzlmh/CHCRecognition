import os
import torch
from torchvision import transforms
from PIL import Image
from network.efficientv2_msa import efficientnetv2_s
from tqdm import tqdm  # Import tqdm
import numpy as np

# Define the model path
model_checkpoint_path = 'checkpoints/xx.pth'

# Define the model
model = efficientnetv2_s(num_classes=626)  # Make sure to replace 'num_classes' with the correct number of classes

# Load the pre-trained weights and ignore specified keys

# Move the model to GPU (if available)
if torch.cuda.is_available():
    model.to('cuda')

# Set the model to evaluation mode
model.eval()

# Define image preprocessing transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),  # Resize the image to (128, 128)
    transforms.Lambda(lambda x: x.expand(3, -1, -1))  # Expand single-channel image to three channels
])

# Define a function to generate predictions
def generate_prediction(model, image_path):
    # Load and preprocess the input image
    input_image = Image.open(image_path)
    input_image = transform(input_image)
    input_image = input_image.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        if torch.cuda.is_available():
            input_image = input_image.to('cuda')

        output = model(input_image)

    # Get the predicted class label and corresponding confidence score
    predicted_label = torch.argmax(output, dim=1).item()
    confidence_score = torch.nn.functional.softmax(output, dim=1)[0, predicted_label].item()

    return predicted_label, confidence_score

# Directory containing test images
test_directory = './data2/test'  # Replace with the path to your test set folder

# Record the number of correctly predicted images and total images
correct_predictions = 0
total_images = 0
total_confidence = 0.0

# Process each class subfolder
for class_folder in tqdm(os.listdir(test_directory), desc='Class Folders', unit='folder'):
    class_folder_path = os.path.join(test_directory, class_folder)

    # List all image files in the subfolder
    image_files = [f for f in os.listdir(class_folder_path) if f.endswith('.jpg')]

    # Process each image
    for image_file in tqdm(image_files, desc='Images', unit='image', leave=True):  # Nested progress bar
        image_path = os.path.join(class_folder_path, image_file)

        # Generate prediction
        predicted_label, confidence_score = generate_prediction(model, image_path)

        if not np.isnan(confidence_score):
            # Use folder name as the true label
            true_label = class_folder
            # Convert true_label to integer format
            true_label = int(true_label)

            # Check if the prediction is correct
            if predicted_label == true_label:
                correct_predictions += 1

            # Accumulate confidence score
            total_confidence += confidence_score

            total_images += 1

        # Print image name, predicted label, true label, and confidence score
        print(f'Image: {image_file}, Predicted Label: {predicted_label}, True Label: {true_label}, Confidence: {confidence_score:.2f}')

        total_images += 1
        print(f"Total tested images: {total_images}, Correct predictions: {correct_predictions}, Total confidence score: {total_confidence}")

# Calculate overall test accuracy
if total_images > 0:
    accuracy = correct_predictions / total_images
    print(f'Total Accuracy: {accuracy * 100:.2f}%')
else:
    print('No images in the test set. Cannot calculate accuracy.')

# Calculate average confidence
if total_images > 0:
    average_confidence = total_confidence / total_images
    print(f'Average Confidence: {average_confidence:.2f}')
else:
    print('No images in the test set. Cannot calculate average confidence.')
