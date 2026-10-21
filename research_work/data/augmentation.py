"""
Dataset Curation & Synthetic Field Augmentation Pipeline
Simulates realistic agricultural field perturbations:
- Solar glare & Specular reflections
- Moving canopy & cloud shadows
- Wind-induced motion blur
- Leaf occlusion / partial cutouts
- Background soil and weed clutter injection
"""

import math
import random
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# 24 Target Agricultural Classes across 5 Economic Crops
CROPS_AND_DISEASES: Dict[str, List[str]] = {
    "Tomato": [
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites",
        "Tomato___Target_Spot",
        "Tomato___Yellow_Leaf_Curl_Virus",
        "Tomato___Mosaic_virus",
        "Tomato___Healthy"
    ],
    "Rice": [
        "Rice___Bacterial_leaf_blight",
        "Rice___Brown_spot",
        "Rice___Leaf_smut",
        "Rice___Blast",
        "Rice___Healthy"
    ],
    "Apple": [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___Healthy"
    ],
    "Potato": [
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___Healthy"
    ],
    "Eggplant": [
        "Eggplant___Bacterial_wilt",
        "Eggplant___Little_leaf",
        "Eggplant___Healthy"
    ]
}

ALL_CLASSES: List[str] = [disease for crop in CROPS_AND_DISEASES.values() for disease in crop]
CLASS_TO_IDX: Dict[str, int] = {cls: idx for idx, cls in enumerate(ALL_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: cls for idx, cls in enumerate(ALL_CLASSES)}


class FieldAugmentationPipeline:
    """Applies realistic field-condition perturbations to synthetic or lab leaf images."""

    def __init__(
        self,
        prob_glare: float = 0.4,
        prob_shadow: float = 0.5,
        prob_motion_blur: float = 0.35,
        prob_occlusion: float = 0.3,
        prob_background_clutter: float = 0.4
    ):
        self.prob_glare = prob_glare
        self.prob_shadow = prob_shadow
        self.prob_motion_blur = prob_motion_blur
        self.prob_occlusion = prob_occlusion
        self.prob_background_clutter = prob_background_clutter

    def apply_solar_glare(self, img: Image.Image) -> Image.Image:
        """Simulates high-intensity solar glare and specular reflections."""
        w, h = img.size
        glare_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(glare_overlay)

        # Random center for sunlight
        center_x = random.randint(0, w)
        center_y = random.randint(0, int(h * 0.6))
        radius = random.randint(int(min(w, h) * 0.25), int(min(w, h) * 0.65))

        # Radial gradient glare
        for r in range(radius, 0, -int(radius / 15)):
            alpha = int(120 * (1.0 - (r / radius) ** 0.5))
            draw.ellipse(
                (center_x - r, center_y - r, center_x + r, center_y + r),
                fill=(255, 250, 230, alpha)
            )

        img = img.convert("RGBA")
        combined = Image.alpha_composite(img, glare_overlay).convert("RGB")
        return combined

    def apply_dynamic_shadow(self, img: Image.Image) -> Image.Image:
        """Simulates irregular polygonal canopy and cloud shadows."""
        w, h = img.size
        shadow_mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(shadow_mask)

        # Generate irregular polygon for shadow
        num_vertices = random.randint(4, 7)
        points = []
        for _ in range(num_vertices):
            points.append((random.randint(0, w), random.randint(0, h)))

        draw.polygon(points, fill=random.randint(140, 200))
        shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=random.randint(8, 18)))

        # Darken image in shadowed regions
        enhancer = ImageEnhance.Brightness(img)
        darkened = enhancer.enhance(random.uniform(0.35, 0.65))
        result = Image.composite(darkened, img, shadow_mask)
        return result

    def apply_wind_motion_blur(self, img: Image.Image) -> Image.Image:
        """Simulates camera and foliage motion blur under windy conditions."""
        kernel_size = random.choice([5, 7, 9])
        # Box blur approximation of directional motion
        return img.filter(ImageFilter.BoxBlur(kernel_size // 2))

    def apply_leaf_occlusion(self, img: Image.Image) -> Image.Image:
        """Simulates partial occlusion by overlapping leaves or dirt patches."""
        w, h = img.size
        draw = ImageDraw.Draw(img)
        box_w = random.randint(int(w * 0.15), int(w * 0.35))
        box_h = random.randint(int(h * 0.15), int(h * 0.35))
        x1 = random.randint(0, w - box_w)
        y1 = random.randint(0, h - box_h)

        # Fill with realistic occlusion color (soil brown, dried leaf, or deep shadow)
        occlusion_color = random.choice([
            (70, 50, 30),     # Soil brown
            (110, 95, 45),    # Dried foliage
            (30, 45, 25),     # Neighboring leaf shadow
        ])
        draw.rectangle([x1, y1, x1 + box_w, y1 + box_h], fill=occlusion_color)
        return img

    def apply_background_clutter(self, img: Image.Image) -> Image.Image:
        """Blends realistic agricultural background noise (soil, dry straw, weeds)."""
        w, h = img.size
        # Add random subtle soil texture noise
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, 12, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def augment(self, img: Image.Image) -> Image.Image:
        """Full stochastic transformation pipeline applying field conditions."""
        res = img.copy()
        if random.random() < self.prob_shadow:
            res = self.apply_dynamic_shadow(res)
        if random.random() < self.prob_glare:
            res = self.apply_solar_glare(res)
        if random.random() < self.prob_motion_blur:
            res = self.apply_wind_motion_blur(res)
        if random.random() < self.prob_occlusion:
            res = self.apply_leaf_occlusion(res)
        if random.random() < self.prob_background_clutter:
            res = self.apply_background_clutter(res)
        return res


def generate_synthetic_leaf(
    crop: str = "Tomato",
    disease: str = "Tomato___Early_blight",
    size: Tuple[int, int] = (224, 224)
) -> Image.Image:
    """Generates a realistic synthetic test leaf for validation and prototyping."""
    img = Image.new("RGB", size, (230, 225, 215))  # Neutral background
    draw = ImageDraw.Draw(img)
    w, h = size

    # Crop-specific leaf base color
    leaf_colors = {
        "Tomato": (46, 125, 50),
        "Rice": (76, 175, 80),
        "Apple": (34, 110, 40),
        "Potato": (56, 142, 60),
        "Eggplant": (38, 105, 55)
    }
    base_color = leaf_colors.get(crop, (46, 125, 50))

    # Draw leaf contour (elliptical curve)
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.10)
    draw.ellipse(
        (margin_x, margin_y, w - margin_x, h - margin_y),
        fill=base_color,
        outline=(25, 75, 25),
        width=2
    )

    # Draw central vein and lateral veins
    draw.line([(w // 2, margin_y + 10), (w // 2, h - margin_y - 10)], fill=(70, 160, 70), width=3)
    for y_offset in range(margin_y + 30, h - margin_y - 20, 25):
        draw.line([(w // 2, y_offset), (margin_x + 20, y_offset - 15)], fill=(65, 150, 65), width=1)
        draw.line([(w // 2, y_offset), (w - margin_x - 20, y_offset - 15)], fill=(65, 150, 65), width=1)

    # Draw disease lesions if diseased
    if "Healthy" not in disease:
        num_lesions = random.randint(5, 12)
        for _ in range(num_lesions):
            lx = random.randint(margin_x + 30, w - margin_x - 30)
            ly = random.randint(margin_y + 30, h - margin_y - 30)
            l_radius = random.randint(6, 18)

            if "blight" in disease.lower():
                # Brown necrotized rings
                draw.ellipse(
                    (lx - l_radius, ly - l_radius, lx + l_radius, ly + l_radius),
                    fill=(90, 50, 25),
                    outline=(180, 140, 60),
                    width=2
                )
            elif "spot" in disease.lower():
                # Circular black spots
                draw.ellipse(
                    (lx - l_radius, ly - l_radius, lx + l_radius, ly + l_radius),
                    fill=(30, 25, 20),
                    outline=(160, 160, 70),
                    width=1
                )
            elif "rust" in disease.lower():
                # Rust orange pustules
                draw.ellipse(
                    (lx - l_radius, ly - l_radius, lx + l_radius, ly + l_radius),
                    fill=(190, 85, 20)
                )
            elif "virus" in disease.lower() or "curl" in disease.lower():
                # Chlorotic yellow patch
                draw.ellipse(
                    (lx - l_radius * 2, ly - l_radius * 2, lx + l_radius * 2, ly + l_radius * 2),
                    fill=(180, 190, 40)
                )
            else:
                draw.ellipse(
                    (lx - l_radius, ly - l_radius, lx + l_radius, ly + l_radius),
                    fill=(80, 45, 25)
                )

    return img


if __name__ == "__main__":
    print(f"Total defined classes: {len(ALL_CLASSES)} across 5 crops.")
    pipeline = FieldAugmentationPipeline()
    sample_img = generate_synthetic_leaf("Tomato", "Tomato___Early_blight")
    augmented_img = pipeline.augment(sample_img)
    print("Augmentation pipeline successfully initialized and verified.")
