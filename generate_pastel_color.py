from PIL import Image, ImageDraw
import random
from colorsys import hls_to_rgb

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

colors = generate_pastel_colors(15, display=True)

for color in colors:
    print(color)