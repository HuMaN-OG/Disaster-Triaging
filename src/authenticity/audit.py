import os
import json
from typing import Dict, Any, List
from PIL import Image

try:
    from .config import AuthenticityConfig
    from .logger import get_logger
except ImportError:
    from src.authenticity.config import AuthenticityConfig
    from src.authenticity.logger import get_logger

logger = get_logger("authenticity.audit")

def audit_authenticity_dataset(config: AuthenticityConfig) -> Dict[str, Any]:
    config.create_dirs()
    logger.info("Starting dataset audit...")
    
    real_counts = {}
    total_real = 0
    corrupted_files = []
    formats = {}
    sizes = []

    for sdir in config.real_subdirs:
        full_dir = os.path.join(config.dataset_dir, sdir)
        count = 0
        if os.path.exists(full_dir):
            for root, _, files in os.walk(full_dir):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
                        path = os.path.join(root, f)
                        count += 1
                        total_real += 1
                        formats[ext] = formats.get(ext, 0) + 1
                        try:
                            with Image.open(path) as img:
                                sizes.append(img.size)
                        except Exception:
                            corrupted_files.append(path)
        real_counts[sdir] = count

    ai_counts = {}
    total_ai = 0
    full_ai_dir = os.path.join(config.dataset_dir, config.ai_subdir)
    if os.path.exists(full_ai_dir):
        for root, dirs, files in os.walk(full_ai_dir):
            rel_path = os.path.relpath(root, full_ai_dir)
            category = rel_path if rel_path != "." else "root"
            count = 0
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
                    path = os.path.join(root, f)
                    count += 1
                    total_ai += 1
                    formats[ext] = formats.get(ext, 0) + 1
                    try:
                        with Image.open(path) as img:
                            sizes.append(img.size)
                    except Exception:
                        corrupted_files.append(path)
            if count > 0:
                ai_counts[category] = count

    total_images = total_real + total_ai
    imbalance_ratio = round(total_real / max(total_ai, 1), 2)
    
    avg_width = int(sum(s[0] for s in sizes) / max(len(sizes), 1)) if sizes else 0
    avg_height = int(sum(s[1] for s in sizes) / max(len(sizes), 1)) if sizes else 0

    audit_result = {
        "total_images": total_images,
        "total_real_images": total_real,
        "total_ai_images": total_ai,
        "imbalance_ratio_real_vs_ai": f"{imbalance_ratio}:1",
        "real_breakdown": real_counts,
        "ai_breakdown": ai_counts,
        "image_formats": formats,
        "average_dimensions": {"width": avg_width, "height": avg_height},
        "corrupted_images_count": len(corrupted_files),
        "corrupted_files": corrupted_files[:10]
    }

    # Save JSON report
    report_json_path = os.path.join(config.data_dir, "dataset_audit_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2)

    # Save Markdown Summary
    summary_md_path = os.path.join(config.data_dir, "dataset_audit_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Audit Summary\n\n")
        f.write(f"- **Total Images**: {total_images}\n")
        f.write(f"- **Real Photographs**: {total_real}\n")
        f.write(f"- **AI-Generated Images**: {total_ai}\n")
        f.write(f"- **Imbalance Ratio (Real:AI)**: {imbalance_ratio}:1\n")
        f.write(f"- **Average Resolution**: {avg_width}x{avg_height}\n")
        f.write(f"- **Corrupted Images**: {len(corrupted_files)}\n\n")
        f.write("### Real Subfolders:\n")
        for k, v in real_counts.items():
            f.write(f"- `{k}`: {v} images\n")
        f.write("\n### AI Subfolders:\n")
        for k, v in ai_counts.items():
            f.write(f"- `{k}`: {v} images\n")

    logger.info(f"Dataset audit complete. Total={total_images} (Real={total_real}, AI={total_ai}). Report saved to {report_json_path}")
    return audit_result

if __name__ == "__main__":
    cfg = AuthenticityConfig()
    audit_authenticity_dataset(cfg)
