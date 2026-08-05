"""
config.py
---------
Centralised configuration management.

Planned implementation
----------------------
- Loads config from a YAML file (config.yaml) at project root.
- Merges with environment variables (via python-dotenv).
- Exposes a typed Config dataclass for use across modules.

Key config fields (planned):
    data:
        raw_dir: data/raw/
        processed_dir: data/processed/
        images_dir: data/images/
        labels_dir: data/labels/
    models:
        artifacts_dir: models/
        text_model: all-MiniLM-L6-v2
        image_model: ViT-B/32
    training:
        seed: 42
        n_folds: 5
        test_size: 0.15
        val_size: 0.15
"""

# TODO: implement Config dataclass
# TODO: implement load_config(path) -> Config
