from PIL import Image
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
import random
from colorsys import hls_to_rgb


def stack_images(base_image_path, mask_image_paths, output_path, colors):
    """
    Stack multiple grayscale mask images onto a base image.

    Parameters:
    base_image_path (str): Path to the base image.
    mask_image_paths (list of str): List of paths to the grayscale mask images.
    output_path (str): Path where the stacked image will be saved.
    colors (list of tuple of int): List of RGBA colors of the segmented objects in the mask images.
    """
    # Open the base image and convert it to RGBA
    base_image = Image.open(base_image_path).convert("RGBA")

    for i, path in enumerate(mask_image_paths):
        # Open the mask image and convert it to RGBA
        mask_image = Image.open(path).convert("RGBA")

        # Create a new image to hold the segmented object
        segmented = Image.new("RGBA", mask_image.size)

        # Get the pixels of the mask image and the segmented image
        mask_pixels = mask_image.load()
        segmented_pixels = segmented.load()

        # For each pixel in the mask image, if the pixel is white (the segmented object),
        # set the pixel of the segmented image to the corresponding color
        for y in range(mask_image.height):
            for x in range(mask_image.width):
                if mask_pixels[x, y] == (255, 255, 255, 255):
                    segmented_pixels[x, y] = colors[i % len(colors)]

        # Composite the base image and the segmented image
        base_image = Image.alpha_composite(base_image, segmented)

    # Save the result to the output path
    base_image.save(output_path)


def convert_jpg_to_png(jpg_path, png_path):
    """
    Convert a JPEG image to a PNG image.

    Parameters:
    jpg_path (str): Path to the JPEG image.
    png_path (str): Path where the PNG image will be saved.
    """
    # Open the JPEG image
    image = Image.open(jpg_path)

    # Save the image as PNG
    image.save(png_path)

def generate_pastel_colors(n, display=False):
    """
    Generate a palette of n pastel colors.

    Parameters:
    n (int): The number of colors to generate.
    display (bool): If True, display the colors.

    Returns:
    list: A list of n RGBA color tuples.
    """
    colors = []

    for i in range(n):
        # Pick a hue in [0, 1), a lightness in [0.7, 0.9), and a saturation in [0.1, 0.3)
        h = i / n
        l = 0.7 + 0.2 * random.random()
        s = 0.1 + 0.2 * random.random()

        # Convert the HSL color to RGB
        r, g, b = hls_to_rgb(h, l, s)

        # Convert the RGB color to RGBA with semi-transparency and add it to the palette
        colors.append((int(r * 255), int(g * 255), int(b * 255), 128))

    if display:
        # Create a new image
        image = Image.new("RGBA", (n * 50, 50))

        # Create a draw object
        draw = ImageDraw.Draw(image)

        # Draw a rectangle for each color
        for i, color in enumerate(colors):
            draw.rectangle([(i * 50, 0), ((i + 1) * 50, 50)], fill=color)

        # Display the image
        image.show()

    return colors

def stack_image_set(stacked_output_path, masked_images_path, source_images_path, random_pick_count=None):
    """
    For each image found in source_images_path directory, convert it to PNG (stored in a temporary location) and then stack with each corresponding PNG mask found in masked_images_path.
    If random_pick_count is provided, only that number of images are picked at random from source_images_path.

    Parameters:
    stacked_output_path (str): Path where the stacked image will be saved.
    masked_images_path (str): List of paths to the grayscale mask images.
    source_images_path (str): Path to the base image.
    random_pick_count (int): Number of images to pick at random from source_images_path. If None, all images are used.
    """

    # Convert source_images_path to Path object for easy file manipulation
    source_images_path = Path(source_images_path)
    masked_images_path = Path(masked_images_path)
    stacked_output_path = Path(stacked_output_path)

    # Create stacked_output_path directory if it does not exist
    stacked_output_path.mkdir(parents=True, exist_ok=True)

    # Get all JPEG images in source_images_path
    all_images = list(source_images_path.glob("*.jpg"))

    # If random_pick_count is provided and less than the total number of images, randomly pick that number of images
    if random_pick_count is not None and random_pick_count < len(all_images):
        images = random.sample(all_images, random_pick_count)
    else:
        images = all_images

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Loop through the selected JPEG images
        for jpg_file in images:
            # Convert JPEG to PNG and save in the temporary directory
            #print(jpg_file)
            png_path = temp_dir / f"{jpg_file.stem}.png"
            convert_jpg_to_png(jpg_file, png_path)
            print(png_path)

            # Find corresponding masks in masked_images_path
            mask_paths = [mask_path for mask_path in masked_images_path.glob(f"{jpg_file.stem}/*.png")] # if mask_path.name != '0.png']

            #print(mask_paths)

            # If no masks were found for this image, continue to the next image
            if not mask_paths:
                continue

            # Define a list of colors for the masks
            colors = generate_pastel_colors(15, display=False)

            # Stack the base image with its masks
            output_path = stacked_output_path / f"{jpg_file.stem}_stacked.png"
            stack_images(png_path, mask_paths, output_path, colors)




stacked_output_path = "/tank0/GaiaData/Malena_stacked/"
masked_images_path = "/tank0/GaiaData/Malena_segmented/"
source_images_path = "/tank0/GaiaData/Malena/"




stack_image_set(stacked_output_path, masked_images_path, source_images_path, 15)

