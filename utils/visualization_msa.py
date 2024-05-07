import cv2
import torch
import os
import numpy as np
from pytorch_grad_cam import GradCAM
from network.efficientv2_msa import efficientnetv2_s

import matplotlib.pyplot as plt


# Custom model class

def main():
    # Load custom model
    model = efficientnetv2_s(num_classes=1000)
    model.load_state_dict(torch.load('checkpoints/cam_test/ca_iter_039.pth'))  # Load model weights
    model.eval()

    # Initialize Grad-CAM
    print(model.blocks)  # Print model layers
    print(model.blocks[39])  # Print the 39th layer of the model
    gradcam = GradCAM(arch=model, target_layer=model.blocks[-1].project_conv.conv)

    # Input image folder path and output folder path
    image_folder = 'cam_test/test'  # Replace with the folder path containing test images
    output_folder = 'cam_test/cam'  # Replace with the folder path to save Grad-CAM results

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in os.listdir(image_folder):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (128, 128))

            # Convert image data to tensor and normalize
            image = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)
            image = image.unsqueeze(0)

            # Get heatmap and heatmap scores
            heatmap, heatmap_scores = gradcam(image)

            # Extract heatmap data
            heatmap_array = heatmap.numpy()

            # Resize heatmap and normalize
            heatmap_resized = cv2.resize(heatmap_array[0, 0], (128, 128))
            heatmap_normalized = heatmap_resized / np.max(heatmap_resized)

            result_image = (image[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            heatmap_overlay = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            result_image = cv2.addWeighted(result_image, 0.5, heatmap_overlay, 0.5, 0)

            # Perform prediction
            with torch.no_grad():
                output = model(image)
                predicted_probs = torch.nn.functional.softmax(output, dim=1)
                predicted_class = torch.argmax(predicted_probs, dim=1).item()
                predicted_prob = predicted_probs[0, predicted_class].item()

            # Display predicted class and probability on the image
            label = f"Class: {predicted_class}, Probability: {predicted_prob:.4f}"
            cv2.putText(result_image, label, (-5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            output_path = os.path.join(output_folder, f'{image_file}_gradcam.jpg')

            # Save Grad-CAM result
            cv2.imwrite(output_path, result_image)
            print(f'Grad-CAM result saved to: {output_path}')


if __name__ == "__main__":
    main()
