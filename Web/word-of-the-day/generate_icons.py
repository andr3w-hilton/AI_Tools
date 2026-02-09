from PIL import Image, ImageDraw, ImageFont
import os

# Create icons directory if it doesn't exist
os.makedirs('icons', exist_ok=True)

# Icon sizes needed for PWA
sizes = [72, 96, 128, 144, 152, 167, 180, 192, 384, 512]

# Create a beautiful gradient icon with "W" for Word
for size in sizes:
    # Create image with gradient background
    img = Image.new('RGB', (size, size), '#667eea')
    draw = ImageDraw.Draw(img)

    # Create gradient effect
    for i in range(size):
        # Gradient from purple to blue
        r = int(102 + (118 - 102) * i / size)
        g = int(126 + (75 - 126) * i / size)
        b = int(234 + (162 - 234) * i / size)
        draw.rectangle([(0, i), (size, i+1)], fill=(r, g, b))

    # Draw a white circle background for the letter
    circle_margin = size // 6
    circle_bbox = [circle_margin, circle_margin, size - circle_margin, size - circle_margin]
    draw.ellipse(circle_bbox, fill='white')

    # Add the letter "W" in the center
    try:
        # Try to use a nice font
        font_size = size // 2
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        # Fallback to default font
        font = ImageFont.load_default()

    # Draw "W" in gradient color
    text = "W"

    # Get text bounding box for centering
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    position = ((size - text_width) // 2, (size - text_height) // 2 - size // 15)

    # Draw text in purple gradient color
    draw.text(position, text, fill='#6b46c1', font=font)

    # Save the icon
    filename = f'icons/icon-{size}.png'
    img.save(filename, 'PNG')
    print(f'Created {filename}')

print('All icons generated successfully!')
