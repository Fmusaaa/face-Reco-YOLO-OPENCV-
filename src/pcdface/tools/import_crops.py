"""Impor crop wajah yang dipotong tangan — `python -m pcdface import-crops` (E9).

Peserta memotong wajahnya sendiri dari foto asli dengan aplikasi apa saja (Preview, Photos,
Paint, Photoshop, …) dan menyimpan hasilnya di `data/crop_manual/` (atau `data/crop-manual/`),
boleh dalam subfolder dan dengan nama berkas apa pun:

    data/DATASET-MUSANG-PCD/S01/007.JPG      ← foto asli (tidak diubah)
    data/crop_manual/Mhs01/001.jpg           ← crop tangan dari foto mana saja di dataset

    python -m pcdface import-crops --input data/DATASET-MUSANG-PCD
    python -m pcdface eval-folder --input data/DATASET-MUSANG-PCD --detectors haar

Foto asal setiap crop ditemukan dari **isi gambarnya**, bukan dari namanya: semua foto di
dataset disaring dua cara — pencocokan pada skala asli crop dan penyaringan multi-skala pada
resolusi kecil — tiga kandidat terbaik dari masing-masing diperiksa teliti, dan pasangan
berskor tertinggi dipilih. Bila crop ada di subfolder bernama kode peserta (`S01/`), hanya foto
peserta itu yang dicari. Identitas peserta diambil dari folder foto asal (`S01`, …) — nama
folder crop tidak dipakai sebagai label.

Letak crop di foto asal dicari dengan *template matching* OpenCV (`cv2.matchTemplate`,
TM_CCOEFF_NORMED) pada beberapa skala — crop yang diperkecil, diperbesar, atau dijadikan
persegi saat disimpan tetap ketemu — dan, bila gagal, pada rotasi 90°/180°/270°, lalu
diperhalus di resolusi asli. Kotak hasil ditulis ke `data/annotations/folder_<nama folder>.json`
(berkas yang sama dengan `annotate --folder`, jadi bisa diperiksa dengan alat itu). Crop dengan
skor di bawah `experiments.e9.manual_crop_min_score` tidak punya foto asal di dataset dan
dilaporkan; dua crop dari wajah yang sama di satu foto disimpan sekali (skor tertinggi).

Laporan per crop: `results/e9_impor_crop.csv` (berkas, foto asal, kotak, skor, skala, status).
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
from pcdface.evaluation.matching import iou
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


SCREEN_SIDE = 320                   # sisi terpanjang foto saat penyaringan cepat
SCREEN_SCALES = np.geomspace(0.15, 4.0, 30)
CANDIDATES = 3                      # foto teratas hasil penyaringan yang diperiksa teliti
DUPLICATE_IOU = 0.5                 # dua crop di foto yang sama dengan IoU ≥ ini = wajah yang sama


def list_crop_files(crops_dir: Path) -> list[Path]:
    """Berkas gambar di `crops_dir` dan subfoldernya (nama apa pun), terurut."""
    return sorted(p for p in crops_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


class PhotoIndex:
    """Semua foto dataset: versi kecil abu-abu untuk penyaringan, foto asli dimuat saat perlu."""

    def __init__(self, folder: Path, keys: list[str]) -> None:
        self.folder = folder
        self.small: dict[str, tuple[np.ndarray, float]] = {}
        self.work: dict[str, tuple[np.ndarray, float]] = {}
        for key in keys:
            image = cv2.imread(str(folder / key), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"foto tidak bisa dibaca: {key}")
            for store, side in ((self.small, SCREEN_SIDE), (self.work, WORK_SIDE)):
                factor = min(1.0, side / max(image.shape))
                store[key] = (cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
                              if factor < 1 else image, factor)

    def screen(self, crop_bgr: np.ndarray, subject: str | None = None) -> list[tuple[float, str]]:
        """Skor kasar setiap foto (hanya foto `subject` bila diberikan), menurun."""
        crop = _gray(crop_bgr)
        scored = [(_search(small, crop, SCREEN_SCALES * factor)[0], key) for key, (small, factor) in self.small.items()
                  if subject is None or key.startswith(f"{subject}/")]
        return sorted(scored, reverse=True)

    def exact(self, crop_bgr: np.ndarray, subject: str | None = None) -> list[tuple[float, str]]:
        """Skor setiap foto bila crop disimpan tanpa diubah ukurannya (skala 1), menurun.

        Lebih tajam daripada `screen` untuk foto beruntun yang hampir sama, karena detail halus
        (derau sensor, tekstur) masih ada di resolusi kerja."""
        rotations = [_rotated(_gray(crop_bgr), angle) for angle in (0, 90, 180, 270)]   # crop tersimpan berputar
        scored = [(max(_search(work, crop, [factor])[0] for crop in rotations), key)
                  for key, (work, factor) in self.work.items() if subject is None or key.startswith(f"{subject}/")]
        return sorted(scored, reverse=True)

    def load(self, key: str) -> np.ndarray:
        return cv2.imread(str(self.folder / key))


@dataclass
class ImportReport:
    rows: list[dict[str, object]]
    accepted_photos: int
    unmatched: list[str]
    duplicates: list[str]
    store: Path


def import_crops(cfg: Config, paths: ProjectPaths, folder: Path, crops_dir: Path,
                 log=lambda message: None) -> ImportReport:
    """Temukan foto asal dan letak setiap crop tangan, lalu tulis kotaknya sebagai anotasi folder."""
    min_score = cfg.experiments.e9.manual_crop_min_score
    keys = list_folder_images(folder)
    for key in keys:
        if not SUBJECT_ID.match(key.split("/", 1)[0]):
            raise ValueError(f"nama folder foto harus kode peserta seperti S01, bukan {key.split('/', 1)[0]!r} "
                             "(nama asli dilarang — CLAUDE.md aturan #5)")
    crops = list_crop_files(crops_dir)
    index = PhotoIndex(folder, keys)
    candidates: dict[str, list[tuple[Located, str]]] = {}
    rows: dict[str, dict[str, object]] = {}
    unmatched: list[str] = []
    for number, path in enumerate(crops, start=1):
        name = path.relative_to(crops_dir).as_posix()
        crop = cv2.imread(str(path))
        if crop is None:
            rows[name] = {"crop": name, "status": "berkas crop rusak"}
            unmatched.append(name)
            continue
        # crop di subfolder kode peserta (S01/…) hanya dicari di foto peserta itu — mencegah pasangan lintas orang
        parts = path.relative_to(crops_dir).parts
        subject = parts[0] if len(parts) > 1 and SUBJECT_ID.match(parts[0]) and (folder / parts[0]).is_dir() else None
        keys_to_check = dict.fromkeys([key for _, key in index.exact(crop, subject)[:CANDIDATES]] +
                                      [key for _, key in index.screen(crop, subject)[:CANDIDATES]])
        options = [(locate_crop(index.load(key), crop), key) for key in keys_to_check]
        candidates[name] = sorted(options, key=lambda item: -item[0].score)
        best, key = candidates[name][0]
        log(f"{number}/{len(crops)} {name} → {key} (skor {best.score:.3f})")

    # Pemasangan rakus: pasangan crop–foto berskor tertinggi lebih dulu; satu wajah di satu foto hanya
    # boleh diambil satu crop. Crop yang foto terbaiknya sudah terpakai (foto beruntun yang hampir sama)
    # pindah ke kandidat berikutnya selama skornya masih ≥ ambang.
    pairs = sorted(((located.score, name, located, key) for name, options in candidates.items()
                    for located, key in options if located.score >= min_score), key=lambda item: -item[0])
    assigned: dict[str, tuple[Located, str]] = {}
    taken: dict[str, list[Box]] = defaultdict(list)
    for score, name, located, key in pairs:
        if name in assigned or any(iou(located.box, box) >= DUPLICATE_IOU for box in taken[key]):
            continue
        assigned[name] = (located, key)
        taken[key].append(located.box)
    duplicates: list[str] = []
    for name, options in candidates.items():
        located, key = assigned.get(name, options[0])
        x, y, w, h = located.box
        row: dict[str, object] = {"crop": name, "foto": key, "x": x, "y": y, "w": w, "h": h,
                                  "skor": round(located.score, 4), "skala": round(located.scale, 3),
                                  "rotasi": located.rotation, "status": "diterima"}
        if name not in assigned:
            if located.score >= min_score:
                row["status"] = f"duplikat (wajah di {key} sudah diambil crop lain)"
                duplicates.append(name)
            else:
                row.update({"foto": "", "status": f"tidak ada foto asal di dataset (skor terbaik "
                                                   f"{located.score:.2f} di {key})"})
                unmatched.append(name)
        rows[name] = row

    store = folder_annotations_path(paths, folder)
    annotations = load_annotations(store)
    for key, boxes in taken.items():
        if boxes:
            annotations[key] = list(boxes)
    found = {key for key, boxes in taken.items() if boxes}
    rows = [rows[name] for name in sorted(rows)]
    save_annotations(store, annotations)
    paths.results.mkdir(parents=True, exist_ok=True)
    with (paths.results / REPORT).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return ImportReport(rows, len(found), sorted(unmatched), sorted(duplicates), store)


def default_crops_dir(paths: ProjectPaths) -> Path:
    """`data/crop_manual`, atau `data/crop-manual` bila hanya itu yang ada."""
    alternative = paths.manual_crops.with_name(paths.manual_crops.name.replace("_", "-"))
    return alternative if not paths.manual_crops.is_dir() and alternative.is_dir() else paths.manual_crops


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Folder foto asli per kode peserta, mis. data/DATASET-MUSANG-PCD")
    parser.add_argument("--crops", default=None,
                        help="Folder crop tangan, boleh bersubfolder (bawaan: data/crop_manual atau data/crop-manual)")


def run(args: argparse.Namespace, cfg: Config, paths: ProjectPaths | None = None) -> int:
    paths = paths or cfg.paths
    folder = Path(args.input)
    crops_dir = Path(args.crops) if args.crops else default_crops_dir(paths)
    for label, path in (("foto", folder), ("crop", crops_dir)):
        if not path.is_dir():
            print(f"[GAGAL] folder {label} tidak ada: {path}", file=sys.stderr)
            return 2
    print(f"Mencari foto asal setiap crop di {crops_dir} (bisa beberapa menit) …", flush=True)
    try:
        report = import_crops(cfg, paths, folder, crops_dir, log=lambda m: print(f"  {m}", flush=True))
    except ValueError as error:
        print(f"[GAGAL] {error}", file=sys.stderr)
        return 1
    accepted = sum(1 for row in report.rows if row.get("status") == "diterima")
    total = len(list_folder_images(folder))
    print(f"\n{len(report.rows)} crop tangan: {accepted} diterima, {len(report.duplicates)} duplikat, "
          f"{len(report.unmatched)} tanpa foto asal")
    print(f"{report.accepted_photos}/{total} foto punya kotak manual → {report.store}")
    for name in report.unmatched:
        print(f"  TANPA FOTO ASAL: {name} — foto aslinya belum ada di {folder}, atau crop sudah diedit")
    if total > report.accepted_photos:
        print(f"  {total - report.accepted_photos} foto belum punya crop tangan — tidak ikut dievaluasi")
    print(f"Laporan per crop: {paths.results / REPORT}")
    print(f"Berikutnya: python -m pcdface eval-folder --input {folder} --detectors haar")
    return 0
