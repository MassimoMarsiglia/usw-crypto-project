from PIL import Image
import os
import math

# folder where your heatmaps are stored
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEATMAP_DIR = os.path.join(BASE_DIR, "../../images/03_features_plotting")
IMAGE_DIR = os.path.join(BASE_DIR, "../../images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# output file
OUTPUT_FILE = os.path.join(IMAGE_DIR, "03_features_technical_3x3_grid.png")

# get all heatmap image files (png, jpg)
image_files = sorted([
    os.path.join(HEATMAP_DIR, f)
    for f in os.listdir(HEATMAP_DIR)
    if f.lower().endswith((".png", ".jpg"))
])

# load images
images = [Image.open(img) for img in image_files]

# determine grid size
cols = 3
rows = math.ceil(len(images) / cols)

# get max width and height of images (to standardize)
img_width = max(img.width for img in images)
img_height = max(img.height for img in images)

# create blank canvas
grid_width = cols * img_width
grid_height = rows * img_height
combined = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

# paste images into the grid
for idx, img in enumerate(images):
    row = idx // cols
    col = idx % cols
    x = col * img_width
    y = row * img_height
    combined.paste(img, (x, y))

# save final image
combined.save(OUTPUT_FILE)
