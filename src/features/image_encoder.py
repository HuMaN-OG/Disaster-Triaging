"""
image_encoder.py
----------------
Placeholder for image embedding extraction using a pretrained CLIP model.

Planned implementation
----------------------
Model:  openai/clip-vit-base-patch32  (512-d image embeddings)
        via the `open-clip-torch` or HuggingFace `transformers` library.

Usage (planned):
    encoder = ImageEncoder(model_name="ViT-B/32")
    embeddings = encoder.encode([PIL.Image.open("flood_photo.jpg")])
    # → np.ndarray of shape (N, 512)

Key design decisions
--------------------
- Accepts PIL Images or file paths as input.
- Images are pre-processed with CLIP's native transform pipeline.
- Model weights are FROZEN – only the vision encoder is used (no text tower).
- Supports batched inference for efficiency.
"""

# TODO: implement ImageEncoder class wrapping open_clip / CLIP
# TODO: add image preprocessing (resize, normalize per CLIP spec)
# TODO: add batch_encode utility with progress bar
# TODO: add save/load embedding cache to data/processed/
