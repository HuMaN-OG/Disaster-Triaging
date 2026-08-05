"""
fusion.py
---------
Placeholder for multimodal feature fusion.

Planned implementation
----------------------
Strategy 1 – Concatenation (baseline):
    fused = np.concatenate([text_emb, image_emb], axis=-1)
    # → shape (N, 384 + 512) = (N, 896)

Strategy 2 – Weighted concatenation:
    fused = np.concatenate([alpha * text_emb, (1-alpha) * image_emb], axis=-1)

Strategy 3 – MLP projection head (optional):
    A small 2-layer MLP maps the 896-d concat vector to a lower-dimensional
    projected space (e.g. 256-d) before feeding into XGBoost.

Usage (planned):
    fused_features = fuse(text_embeddings, image_embeddings, strategy="concat")
    # → np.ndarray of shape (N, D)
"""

# TODO: implement fuse() function with strategy parameter
# TODO: implement optional MLP projection head (PyTorch)
# TODO: add unit tests for output shape assertions
