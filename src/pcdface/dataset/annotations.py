"""Baca/tulis `boxes.json` — kotak wajah manual (ground truth).

Format:

    {
      "version": 1,
      "annotations": {
        "jarak/S03/jarak_S03_150cm_normal_04.jpg": [[512, 180, 120, 158]],
        "kosong/kosong_07.jpg": []
      }
    }

Kunci adalah path relatif terhadap `data/raw/`. Daftar kosong berarti citra
sudah dianotasi dan memang tidak memuat wajah — berbeda dari kunci yang tidak
ada (belum dianotasi). Tidak ada medan nama atau identitas.

Foto lepas per peserta (`annotate --folder`, E9) memakai format yang sama di berkas
terpisah `folder_<nama folder>.json`, dengan kunci relatif terhadap folder itu
(`S01/001.jpg`).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from pcdface.boxes import Box, to_box

if TYPE_CHECKING:
    from pcdface.paths import ProjectPaths

FORMAT_VERSION = 1
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def folder_annotations_path(paths: ProjectPaths, folder: Path) -> Path:
    """Berkas anotasi untuk foto lepas di `folder`, di samping `boxes.json`."""
    return paths.annotations.parent / f"folder_{folder.resolve().name}.json"


def list_folder_images(folder: Path) -> list[str]:
    """Foto di `folder/<kode peserta>/` sebagai kunci relatif (`S01/001.jpg`), terurut."""
    return sorted(p.relative_to(folder).as_posix() for p in folder.glob("*/*")
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def load_annotations(path: Path) -> dict[str, list[Box]]:
    """Baca anotasi. Berkas yang belum ada dianggap kosong."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    raw = payload.get("annotations", payload) if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: format anotasi tidak dikenal")
    annotations: dict[str, list[Box]] = {}
    for key, boxes in raw.items():
        if not isinstance(boxes, list):
            raise ValueError(f"{path}: nilai untuk {key} harus daftar kotak")
        annotations[key] = [to_box(box) for box in boxes]
    return annotations


def save_annotations(path: Path, annotations: dict[str, list[Box]]) -> None:
    """Tulis anotasi secara atomik, kunci terurut agar diff mudah dibaca."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": FORMAT_VERSION,
        "catatan": "Ground truth deteksi wajah: hanya kotak [x, y, w, h], tanpa identitas.",
        "annotations": {key: [list(box) for box in annotations[key]] for key in sorted(annotations)},
    }
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=1, ensure_ascii=False)
            stream.write("\n")
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
