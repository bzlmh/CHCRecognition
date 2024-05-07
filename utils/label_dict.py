import os

# Specify the parent folder path containing subfolders
parent_folder_path = './data2/train_data'

# Get all subfolders within the parent folder
subfolders = [f for f in os.listdir(parent_folder_path) if os.path.isdir(os.path.join(parent_folder_path, f))]

# Create a dictionary to store the mapping between Chinese labels and numeric labels
label_mapping = {}

# Assign a numeric label to each subfolder
for index, subfolder in enumerate(subfolders):
    label_mapping[index] = subfolder

# Write the mapping between Chinese labels and numeric labels to a txt file
output_file_path = 'label_mapping.txt'
with open(output_file_path, 'w', encoding='utf-8') as file:
    for index, label in label_mapping.items():
        file.write(f'{index}\t{label}\n')

print(f'Label mapping has been written to {output_file_path}')
