from PIL import Image
import os
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEATMAP_DIR = os.path.join(BASE_DIR, "../../images/02_heatmap_correlation")
IMAGE_DIR = os.path.join(BASE_DIR, "../../images")
os.makedirs(IMAGE_DIR, exist_ok=True)

def merge_group(search_key: str, output_name: str, cols: int = 3) -> None:
    """
    Merge all heatmaps whose filename contains `search_key` into a grid image.

    search_key:   e.g. "BTCpairs" or "USDTpairs"
    output_name:  e.g. "02_BTCpairs_heatmaps_grid.png"
    """
    image_files = sorted(
        os.path.join(HEATMAP_DIR, f)
        for f in os.listdir(HEATMAP_DIR)
        if f.lower().endswith((".png", ".jpg")) and search_key.lower() in f.lower()
    )

    if not image_files:
        print(f"No images found for key '{search_key}'. Skipping.")
        return

    images = [Image.open(img) for img in image_files]

    rows = math.ceil(len(images) / cols)
    img_width = max(img.width for img in images)
    img_height = max(img.height for img in images)

    grid_width = cols * img_width
    grid_height = rows * img_height
    combined = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * img_width
        y = row * img_height
        combined.paste(img, (x, y))

    out_path = os.path.join(IMAGE_DIR, output_name)
    combined.save(out_path)
    print(f"Saved merged grid → {out_path}")


def main():
    # BTC-Heatmaps mergen
    merge_group(search_key="BTCpairs",  output_name="02_BTCpairs_heatmaps_grid.png")

    # USDT-Heatmaps mergen
    merge_group(search_key="USDTPairs", output_name="02_USDTpairs_heatmaps_grid.png")
    # falls deine Files USDTpairs heißen (kleines p), würde das dank .lower() trotzdem funktionieren


if __name__ == "__main__":
    main()
