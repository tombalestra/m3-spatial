from PIL import Image, ImageDraw, ImageFont
import numpy as np


def add_image_marker(image_array, text="Image1", font_size=40, padding=10):
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array)
    
    # Create a drawing object
    draw = ImageDraw.Draw(image)
    
    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Get text size
    text_width, text_height = draw.textsize(text, font=font)
    
    # Calculate position for the marker (top right corner)
    x = image.width - text_width - padding * 2
    y = 0
    
    # Draw black rectangle
    draw.rectangle([x, y, image.width, text_height + padding * 2], fill="black")
    
    # Draw white text
    draw.text((x + padding, y + padding), text, font=font, fill="white")
    
    # Convert back to numpy array
    return np.array(image)