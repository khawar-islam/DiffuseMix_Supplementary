import random
from torchvision import datasets, models, transforms
from nets import *
import time, os, copy, argparse
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image as PILImage
from accelerate import PartialState
from accelerate import Accelerator
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

accelerator = Accelerator()

model_id = "timbrooks/instruct-pix2pix"
pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16,
                                                                  use_safetensors=True,
                                                                  safety_checker=None)
distributed_state = PartialState()
pipeline.to(distributed_state.device)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

def generate_images_online(prompt, img_path, num_images_to_generate, guidance_scale=4):
    image = PILImage.open(img_path).convert('RGB').resize((256, 256))
    return pipeline(prompt, image=image, num_images_per_prompt=num_images_to_generate,
                    guidance_scale=guidance_scale).images


# Step 1: Load all the fractal images from a specified directory into a list
fractal_img_dir = "/media/cvpr/CM_1/cvpr2023/fractal_dataset/img"  # Assuming this directory contains the fractal dataset
fractal_img_paths = [os.path.join(fractal_img_dir, fname) for fname in os.listdir(fractal_img_dir) if
                     fname.endswith(('.png', '.jpg', '.jpeg'))]
fractal_imgs = [Image.open(path).convert('RGB').resize((256, 256))  for path in fractal_img_paths]


def rand_bbox(size, lam):
    W, H = size
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)  # Use Python's built-in int type
    cut_h = int(H * cut_rat)  # Use Python's built-in int type

    # Uniformly random position for the crop
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class AugmentedDataset(Dataset):

    # prompts = ["Autumn", "snowy", "sunset", "watercolor art","rain", "rainbow", "aurora",
    #            "mosaic", "ukiyo-e", "a sketch with crayon"]

    prompts = ["Autumn", "snowy", "sunset", "ukiyo-e"]
    #prompts = ["Autumn"]



    def __init__(self, original_dataset, num_augmented_images_per_image, guidance_scale=4):
        self.original_dataset = original_dataset
        self.combine_counter = 0
        self.num_augmented_images_per_image = num_augmented_images_per_image
        self.guidance_scale = guidance_scale
        self.augmented_images = self.generate_augmented_images()

    @staticmethod
    def blend_images_with_resize(base_img, overlay_img, alpha=0.20):

        overlay_img_resized = overlay_img.resize(base_img.size)

        # Convert images to numpy arrays for easier processing
        base_array = np.array(base_img, dtype=np.float32)
        overlay_array = np.array(overlay_img_resized, dtype=np.float32)

        # Ensure the arrays have shape (height, width, 3)
        assert base_array.shape == overlay_array.shape and len(base_array.shape) == 3

        # Blend the images
        blended_array = (1 - alpha) * base_array + alpha * overlay_array
        blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        blended_img = Image.fromarray(blended_array)
        return blended_img

    def combine_images(self, original_img, generated_img, beta=1.0):
        """
        Replace a random region of the original image with a region from the generated image,
        following the CutMix style.
        """
        width, height = original_img.size
        lam = np.random.beta(beta, beta)
        bbx1, bby1, bbx2, bby2 = rand_bbox((width, height), lam)

        patch = generated_img.crop((bbx1, bby1, bbx2, bby2))
        combined_img = original_img.copy()
        combined_img.paste(patch, (bbx1, bby1))

        return combined_img

    def generate_augmented_images(self):
        augmented_data = []  # This will store (image, label) tuples

        # Define base directory for saving images
        base_directory = './base_directory'
        original_resized_dir = os.path.join(base_directory, 'original_resized')
        generated_dir = os.path.join(base_directory, 'generated')
        fractal_dir = os.path.join(base_directory, 'fractal')
        concatenated_dir = os.path.join(base_directory, 'concatenated')
        blended_dir = os.path.join(base_directory, 'blended')

        # Ensure these directories exist
        os.makedirs(original_resized_dir, exist_ok=True)
        os.makedirs(generated_dir, exist_ok=True)
        os.makedirs(fractal_dir, exist_ok=True)
        os.makedirs(concatenated_dir, exist_ok=True)
        os.makedirs(blended_dir, exist_ok=True)

        for idx, (img_path, label) in enumerate(self.original_dataset.samples):

            # Create subdirectories for the label within each type of image directory
            label_original_resized_dir = os.path.join(original_resized_dir, str(label))
            label_generated_dir = os.path.join(generated_dir, str(label))
            label_fractal_dir = os.path.join(fractal_dir, str(label))
            label_concatenated_dir = os.path.join(concatenated_dir, str(label))
            label_blended_dir = os.path.join(blended_dir, str(label))

            # Ensure these label directories exist
            os.makedirs(label_original_resized_dir, exist_ok=True)
            os.makedirs(label_generated_dir, exist_ok=True)
            os.makedirs(label_fractal_dir, exist_ok=True)
            os.makedirs(label_concatenated_dir, exist_ok=True)
            os.makedirs(label_blended_dir, exist_ok=True)


            original_img = PILImage.open(img_path).convert('RGB')
            original_img = original_img.resize((256, 256))
            img_filename = os.path.basename(img_path)

            # 1. Save original image after resize
            #original_img.save(os.path.join(original_resized_dir, img_filename))
            original_img.save(os.path.join(label_original_resized_dir, img_filename))

            # For each prompt, generate augmented images
            for prompt in self.prompts:
                description = f"{prompt} version of {img_filename}"
                print(f"Generating images for prompt: {description}")

                for i, img in enumerate(
                        generate_images_online(description, img_path, self.num_augmented_images_per_image,
                                               self.guidance_scale)):
                    img = img.resize((256, 256))

                    # 2. Save generated image
                    generated_img_filename = f"{img_filename}_generated_{prompt}_{i}.jpg"
                    img.save(os.path.join(label_generated_dir, generated_img_filename))


                    if not self.is_black_image(img):
                        combined_img = self.combine_images(original_img, img)

                        # 4. Save concatenated image
                        concatenated_img_filename = f"{img_filename}_concatenated_{prompt}_{i}.jpg"
                        combined_img.save(os.path.join(label_concatenated_dir, concatenated_img_filename))

                        # 3. Save the fractal image
                        random_fractal_img = random.choice(fractal_imgs)
                        fractal_img_filename = f"{img_filename}_fractal_{prompt}_{i}.jpg"
                        random_fractal_img.save(os.path.join(label_fractal_dir, fractal_img_filename))

                        blended_img = self.blend_images_with_resize(combined_img, random_fractal_img)

                        # 5. Save the blended image
                        blended_img_filename = f"{img_filename}_blended_{prompt}_{i}.jpg"
                        blended_img.save(os.path.join(label_blended_dir, blended_img_filename))

                        augmented_data.append((blended_img, label))

        return augmented_data

    @staticmethod
    def is_black_image(image):
        histogram = image.convert("L").histogram()
        return histogram[-1] > 0.9 * image.size[0] * image.size[1] and max(histogram[:-1]) < 0.1 * image.size[0] * \
            image.size[1]

    def __len__(self):
        return len(self.augmented_images)

    def __getitem__(self, idx):
        image, label = self.augmented_images[idx]  # Get the image and its associated label
        return image, label


# Load the original dataset
train_directory = '/media/cvpr/CM_1/datasets/flower_data/train'
original_train_dataset = datasets.ImageFolder(root=train_directory)

# Create the augmented dataset
augmented_train_dataset = AugmentedDataset(original_train_dataset, num_augmented_images_per_image=1)