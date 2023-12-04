from PIL import Image
import os


def resize_image(input_image_path, output_image_path, new_size):
    with Image.open(input_image_path) as image:
        # Resize the image and convert to RGB
        resized_image = image.convert('RGB').resize(new_size)
        resized_image.save(output_image_path)


folder_path = '/home/cvpr/Documents/birds'
new_size = (256, 256)  # New width and height

for file_name in os.listdir(folder_path):
    if file_name.endswith('.jpg') or file_name.endswith('.png'):  # Add other file types if needed
        input_path = os.path.join(folder_path, file_name)
        output_path = os.path.join(folder_path, 'resized_' + file_name)
        resize_image(input_path, output_path, new_size)
