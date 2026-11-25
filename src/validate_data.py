from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

EXPECTED_SPLITS = {"train": 1034, "validation": 133, "test": 128}
EXPECTED_CLASSES = {"angular_leaf_spot", "bean_rust", "healthy"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _decode_fully(path: Path) -> None:
    """Read every pixel, not just the header, so truncated files are caught.

    Pillow's open() only parses the header; load() decodes the pixel data.
    Errors are re-raised as OSError (the parent of UnidentifiedImageError)
    with the offending path in the message.
    """
    try:
        with Image.open(path) as image:
            image.load()
    except OSError as exc:
        raise OSError(f"Cannot decode image {path}: {exc}") from exc


def validate_dataset(raw_root: Path) -> dict:
    """Validate the raw dataset and return a deterministic quality report.

    Core checks:
      1. required splits exist;
      2. exactly the expected classes exist in every split;
      3. expected image counts are present;
      4. every image can be fully decoded by Pillow (`Image.load()`);
      5. no split or class is empty.

    Any failure raises (FileNotFoundError, RuntimeError or OSError) so that
    the DVC stage stops. The report has no timestamps and every list is
    sorted, so the same files always produce the same report.
    Files without an image extension are not counted; they are listed in
    `ignored_files` so they are visible instead of silently skipped.
    """
    raw_root = Path(raw_root)
    # Read the module-level contract at call time (tests patch EXPECTED_SPLITS).
    expected_splits = dict(EXPECTED_SPLITS)
    expected_classes = sorted(EXPECTED_CLASSES)

    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw data folder not found: {raw_root}")

    splits: dict[str, dict] = {}
    ignored_files: list[str] = []

    for split in sorted(expected_splits):
        split_dir = raw_root / split
        # 1. split exists
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")

        # 2. exactly the expected classes
        found_classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        if found_classes != expected_classes:
            missing = sorted(set(expected_classes) - set(found_classes))
            extra = sorted(set(found_classes) - set(expected_classes))
            raise RuntimeError(
                f"Split '{split}' has wrong classes: missing={missing}, unexpected={extra}"
            )

        class_counts: dict[str, int] = {}
        for class_name in expected_classes:
            images: list[Path] = []
            for path in sorted((split_dir / class_name).rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(path)
                else:
                    ignored_files.append(path.relative_to(raw_root).as_posix())

            # 5. no empty class
            if not images:
                raise RuntimeError(f"Class '{class_name}' in split '{split}' has no images")

            # 4. every image decodes completely
            for path in images:
                _decode_fully(path)

            class_counts[class_name] = len(images)

        # 3. exact count per split
        total = sum(class_counts.values())
        if total != expected_splits[split]:
            raise RuntimeError(
                f"Split '{split}' has {total} images, expected {expected_splits[split]}"
            )

        splits[split] = {"images": total, "class_counts": class_counts}

    return {
        "status": "PASS",
        "classes": expected_classes,
        "splits": splits,
        "total_images": sum(s["images"] for s in splits.values()),
        "ignored_files": sorted(ignored_files),
    }


def main() -> None:
    report = validate_dataset(Path("data/raw"))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
