"""Impor crop wajah yang dipotong tangan — `python -m pcdface import-crops` (E9).

Peserta memotong wajahnya sendiri dari foto asli dengan aplikasi apa saja (Preview, Photos,
Paint, Photoshop, …) lalu menyimpannya dengan nama foto asal:

    data/DATASET-MUSANG-PCD/S01/001.JPG      ← foto asli (tidak diubah)
    data/crop_manual/S01/001.jpg             ← crop tangan wajah di foto itu
    data/crop_manual/S01/001_2.jpg           ← wajah kedua di foto yang sama (akhiran _<apa saja>)

    python -m pcdface import-crops --input data/DATASET-MUSANG-PCD
    python -m pcdface eval-folder --input data/DATASET-MUSANG-PCD --detectors haar

Supaya crop tangan bisa dibandingkan dengan crop detektor (IoU, geser, rasio ukuran), posisinya
di foto asli harus diketahui. Posisi itu dicari dengan *template matching* OpenCV
(`cv2.matchTemplate`, TM_CCOEFF_NORMED) pada beberapa skala — crop yang ikut diperkecil atau
diperbesar saat disimpan tetap ketemu — dan, bila gagal, pada rotasi 90°/180°/270°. Kotak hasil
ditulis ke `data/annotations/folder_<nama folder>.json` (berkas yang sama dengan `annotate
--folder`, jadi bisa diperiksa atau dirapikan dengan alat itu). Crop dengan skor di bawah
`experiments.e9.manual_crop_min_score` ditolak dan dilaporkan.

Laporan per crop: `results/e9_impor_crop.csv` (berkas, foto, kotak, skor, skala, status).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pcdface.boxes import Box
from pcdface.config import Config
from pcdface.dataset.annotations import (
    IMAGE_SUFFIXES,
    folder_annotations_path,
    list_folder_images,
    load_annotations,
    save_annotations,
)
from pcdface.dataset.metadata import SUBJECT_ID
from pcdface.paths import ProjectPaths

WORK_SIDE = 1280                    # sisi terpanjang foto saat pencarian (cukup tepat, ±3 px di foto 4000 px)
COARSE_SCALES = np.geomspace(0.25, 4.0, 41)
FINE_STEPS = np.linspace(0.94, 1.06, 13)
EXACT_ENOUGH = 0.97                 # skor pada skala 1 yang dianggap pasti (crop tanpa diubah ukurannya)
CONFIDENT = 0.9                     # di atas ini rotasi lain tidak dicoba lagi
MIN_TEMPLATE_PX = 12
REPORT = "e9_impor_crop.csv"
COLUMNS = ("crop", "foto", "x", "y", "w", "h", "skor", "skala", "rotasi", "status")


@dataclass
class Located:
    box: Box            # posisi crop di foto asli (piksel foto asli)
    score: float        # TM_CCOEFF_NORMED terbaik, −1…1
    scale: float        # ukuran di foto ÷ ukuran berkas crop
    rotation: int       # derajat rotasi crop yang paling cocok (0 bila tidak diputar)


def _gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _search(photo: np.ndarray, crop: np.ndarray, scales) -> tuple[float, float, tuple[int, int]]:
    """(skor, skala, posisi kiri-atas) terbaik di antara `scales` (skala relatif terhadap `photo`)."""
    best = (-2.0, 1.0, (0, 0))
    height, width = photo.shape
    for scale in scales:
        w, h = int(round(crop.shape[1] * scale)), int(round(crop.shape[0] * scale))
        if w < MIN_TEMPLATE_PX or h < MIN_TEMPLATE_PX or w > width or h > height:
            continue
        template = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        if float(template.std()) < 1e-3:
            continue            # template polos — skor tidak terdefinisi
        result = cv2.matchTemplate(photo, template, cv2.TM_CCOEFF_NORMED)
        _, value, _, location = cv2.minMaxLoc(result)
        if value > best[0]:
            best = (float(value), float(scale), location)
    return best


def locate_crop(photo_bgr: np.ndarray, crop_bgr: np.ndarray) -> Located:
    """Cari letak crop di foto: skala 1 dulu, lalu pindai skala kasar → halus, lalu rotasi."""
    photo = _gray(photo_bgr)
    factor = min(1.0, WORK_SIDE / max(photo.shape))
    small = cv2.resize(photo, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA) if factor < 1 else photo
    best: tuple[float, float, tuple[int, int], int] | None = None
    for rotation in (0, 90, 180, 270):
        crop = _rotated(_gray(crop_bgr), rotation)
        score, scale, location = _search(small, crop, [factor])         # crop tanpa diubah ukurannya
        if score < EXACT_ENOUGH:
            coarse = _search(small, crop, COARSE_SCALES * factor)
            if coarse[0] > score:
                score, scale, location = coarse
            fine = _search(small, crop, scale * FINE_STEPS)
            if fine[0] > score:
                score, scale, location = fine
        if best is None or score > best[0]:
            best = (score, scale, location, rotation)
        if best[0] >= CONFIDENT:
            break                # rotasi lain hanya dicoba bila tanpa rotasi belum jelas cocok
    score, scale, (x, y), rotation = best  # type: ignore[misc]
    crop = _rotated(_gray(crop_bgr), rotation)
    # perhalus di resolusi asli: pencarian di foto kecil hanya tepat ±1/factor piksel
    full_scale = scale / factor
    margin = int(np.ceil(2.0 / factor)) + 2
    w, h = int(round(crop.shape[1] * full_scale)), int(round(crop.shape[0] * full_scale))
    x0, y0 = max(int(round(x / factor)) - margin, 0), max(int(round(y / factor)) - margin, 0)
    x1, y1 = min(x0 + w + 2 * margin + 2, photo.shape[1]), min(y0 + h + 2 * margin + 2, photo.shape[0])
    refined = _search(photo[y0:y1, x0:x1], crop, full_scale * np.array([0.99, 1.0, 1.01]))
    if refined[0] >= score:
        score, full_scale, (dx, dy) = refined
        x_full, y_full = x0 + dx, y0 + dy
    else:
        x_full, y_full = int(round(x / factor)), int(round(y / factor))
    box = (int(x_full), int(y_full), int(round(crop.shape[1] * full_scale)), int(round(crop.shape[0] * full_scale)))
    return Located(box, score, full_scale, rotation)


def _rotated(crop: np.ndarray, rotation: int) -> np.ndarray:
    if not rotation:
        return crop
    return cv2.rotate(crop, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                             270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation])


def pair_crops(crops_dir: Path, folder: Path) -> tuple[dict[str, list[Path]], list[str]]:
    """{kunci foto: [berkas crop]} dan daftar crop yang tidak punya foto asal.

    Crop `S01/001.jpg` atau `S01/001_<apa saja>.jpg` milik foto `S01/001.<ekstensi apa saja>`."""
    photos: dict[tuple[str, str], str] = {}
    for key in list_folder_images(folder):
        subject, name = key.split("/", 1)
        photos[(subject, Path(name).stem.lower())] = key
    paired: dict[str, list[Path]] = defaultdict(list)
    orphans: list[str] = []
    for path in sorted(p for p in crops_dir.glob("*/*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
        subject, stem = path.parent.name, path.stem.lower()
        if not SUBJECT_ID.match(subject):
            raise ValueError(f"nama folder crop harus kode peserta seperti S01, bukan {subject!r} "
                             "(nama asli dilarang — CLAUDE.md aturan #5)")
        key = photos.get((subject, stem))
        if key is None and "_" in stem:
            key = photos.get((subject, stem.rsplit("_", 1)[0]))
        if key is None:
            orphans.append(path.relative_to(crops_dir).as_posix())
        else:
            paired[key].append(path)
    return dict(paired), orphans


@dataclass
class ImportReport:
    rows: list[dict[str, object]]
    accepted_photos: int
    rejected: list[str]
    orphans: list[str]
    store: Path


def import_crops(cfg: Config, paths: ProjectPaths, folder: Path, crops_dir: Path) -> ImportReport:
    """Temukan setiap crop tangan di foto aslinya dan tulis kotaknya sebagai anotasi folder."""
    min_score = cfg.experiments.e9.manual_crop_min_score
    paired, orphans = pair_crops(crops_dir, folder)
    store = folder_annotations_path(paths, folder)
    annotations = load_annotations(store)
    rows, rejected, accepted = [], [], 0
    for key in sorted(paired):
        photo = cv2.imread(str(folder / key))
        if photo is None:
            raise ValueError(f"foto tidak bisa dibaca: {key}")
        boxes, ok = [], True
        for path in paired[key]:
            crop = cv2.imread(str(path))
            name = path.relative_to(crops_dir).as_posix()
            if crop is None:
                rows.append({"crop": name, "foto": key, "status": "berkas crop rusak"})
                rejected.append(name)
                ok = False
                continue
            found = locate_crop(photo, crop)
            good = found.score >= min_score
            x, y, w, h = found.box
            rows.append({"crop": name, "foto": key, "x": x, "y": y, "w": w, "h": h, "skor": round(found.score, 4),
                         "skala": round(found.scale, 3), "rotasi": found.rotation,
                         "status": "diterima" if good else f"ditolak (skor < {min_score:g})"})
            if good:
                boxes.append(found.box)
            else:
                rejected.append(name)
                ok = False
        if ok:
            annotations[key] = boxes        # foto dengan crop gagal tidak ditulis: periksa lalu impor ulang
            accepted += 1
        else:
            annotations.pop(key, None)
    save_annotations(store, annotations)
    paths.results.mkdir(parents=True, exist_ok=True)
    with (paths.results / REPORT).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return ImportReport(rows, accepted, rejected, orphans, store)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Folder foto asli per kode peserta, mis. data/DATASET-MUSANG-PCD")
    parser.add_argument("--crops", default=None, help="Folder crop tangan per kode peserta (bawaan: data/crop_manual)")


def run(args: argparse.Namespace, cfg: Config, paths: ProjectPaths | None = None) -> int:
    paths = paths or cfg.paths
    folder = Path(args.input)
    crops_dir = Path(args.crops) if args.crops else paths.manual_crops
    for label, path in (("foto", folder), ("crop", crops_dir)):
        if not path.is_dir():
            print(f"[GAGAL] folder {label} tidak ada: {path}", file=sys.stderr)
            return 2
    try:
        report = import_crops(cfg, paths, folder, crops_dir)
    except ValueError as error:
        print(f"[GAGAL] {error}", file=sys.stderr)
        return 1
    total = len(list_folder_images(folder))
    print(f"{len(report.rows)} crop tangan → {report.accepted_photos}/{total} foto punya kotak manual "
          f"({report.store})")
    for name in report.rejected:
        print(f"  DITOLAK: {name} — tidak ketemu di foto asalnya (diputar, diedit, atau salah nama?)")
    for name in report.orphans:
        print(f"  TANPA FOTO ASAL: {name} — beri nama sama dengan foto, mis. S01/001.jpg atau S01/001_2.jpg")
    if total > report.accepted_photos:
        print(f"  {total - report.accepted_photos} foto belum punya crop tangan — tidak ikut dievaluasi")
    print(f"Laporan per crop: {paths.results / REPORT}")
    print(f"Berikutnya: python -m pcdface eval-folder --input {folder} --detectors haar")
    return 0 if not report.rejected else 1
