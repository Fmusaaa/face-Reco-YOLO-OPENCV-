"""Gambar kotak wajah manual — ground truth seluruh evaluasi (PRD §7).

ATURAN KOTAK (tetap, tulis di Metodologi): batas atas garis tumbuh rambut,
batas bawah ujung dagu, kiri-kanan tepi pipi, tanpa telinga dan leher.

    python -m pcdface annotate                 # mulai dari citra pertama yang belum dianotasi
    python -m pcdface annotate --set multi     # hanya satu set
    python -m pcdface annotate --folder data/DATASET-MUSANG-PCD   # foto lepas per peserta (E9)

Mouse : seret kiri = kotak baru, klik kanan = hapus kotak di bawah kursor
Tombol: n / → simpan & berikutnya     p / ← simpan & sebelumnya
        g     lompat ke yang belum dianotasi
        z     zoom 1× → 2× → 4× di posisi kursor (wajah jauh)
        u     batalkan kotak terakhir     c  hapus semua kotak
        s     simpan sekarang             q / ESC  simpan & keluar

Pindah citra menyimpan kotak citra itu — termasuk daftar kosong, yang berarti
"sudah diperiksa, tidak ada wajah" (untuk set kosong). `validate` menangkap
jumlah kotak yang tidak sama dengan expected_faces.

Mode `--folder` (E9): foto lepas berukuran apa saja di `<folder>/S01/*.jpg`, … tanpa
metadata. Kotak disimpan ke `data/annotations/folder_<nama folder>.json` dengan kunci
path relatif terhadap folder (mis. `S01/001.jpg`). Jendela diperkecil otomatis supaya
foto ponsel muat di layar; koordinat kotak tetap dalam piksel foto asli.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pcdface.boxes import Box, sort_left_to_right
from pcdface.config import Config
from pcdface.dataset.annotations import folder_annotations_path, list_folder_images, load_annotations, save_annotations
from pcdface.dataset.metadata import read_metadata
from pcdface.paths import ProjectPaths
from pcdface.tools.keys import lower_key

WINDOW = "Anotasi - pcdface"
ZOOM_LEVELS = (1, 2, 4)
MIN_DRAG_PX = 3
FIT_WINDOW = (1400, 860)   # batas jendela mode --folder sebelum dikali --display-scale
# Kode panah dari waitKeyEx: 2/3 (beberapa backend), macOS 63234/63235, Linux 65361/65363,
# Windows 2424832/2555904. Kode 81/83 sengaja tidak dipakai — sama dengan 'Q'/'S' saat Caps Lock.
KEY_LEFT = {2, 63234, 65361, 2424832}
KEY_RIGHT = {3, 63235, 65363, 2555904}


@dataclass
class Viewport:
    """Pemetaan koordinat jendela ↔ citra untuk zoom dan skala tampilan."""

    width: int
    height: int
    zoom: int = 1
    center: tuple[float, float] = (0.0, 0.0)
    display_scale: float = 1.0

    @property
    def origin(self) -> tuple[float, float]:
        view_w, view_h = self.width / self.zoom, self.height / self.zoom
        x0 = min(max(self.center[0] - view_w / 2, 0.0), self.width - view_w)
        y0 = min(max(self.center[1] - view_h / 2, 0.0), self.height - view_h)
        return x0, y0

    def to_image(self, wx: float, wy: float) -> tuple[int, int]:
        x0, y0 = self.origin
        scale = self.zoom * self.display_scale
        x = int(round(x0 + wx / scale))
        y = int(round(y0 + wy / scale))
        return min(max(x, 0), self.width), min(max(y, 0), self.height)

    def to_window(self, ix: float, iy: float) -> tuple[int, int]:
        x0, y0 = self.origin
        scale = self.zoom * self.display_scale
        return int(round((ix - x0) * scale)), int(round((iy - y0) * scale))

    def cycle_zoom(self, at_image: tuple[float, float]) -> None:
        self.zoom = ZOOM_LEVELS[(ZOOM_LEVELS.index(self.zoom) + 1) % len(ZOOM_LEVELS)]
        self.center = at_image

    def render(self, image: np.ndarray) -> np.ndarray:
        x0, y0 = self.origin
        view_w, view_h = int(self.width / self.zoom), int(self.height / self.zoom)
        crop = image[int(y0):int(y0) + view_h, int(x0):int(x0) + view_w]
        size = (int(self.width * self.display_scale), int(self.height * self.display_scale))
        # piksel tajam hanya saat benar-benar diperbesar; foto ponsel yang di-zoom masih diperkecil
        interpolation = cv2.INTER_NEAREST if self.zoom * self.display_scale > 1 else cv2.INTER_AREA
        return cv2.resize(crop, size, interpolation=interpolation)


@dataclass
class AnnotationState:
    """Kotak pada citra aktif dan status seret mouse (koordinat citra)."""

    boxes: list[Box] = field(default_factory=list)
    drag_start: tuple[int, int] | None = None
    cursor: tuple[int, int] = (0, 0)

    def finish_drag(self, end: tuple[int, int]) -> Box | None:
        if self.drag_start is None:
            return None
        (x1, y1), (x2, y2) = self.drag_start, end
        self.drag_start = None
        box = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        if box[2] < MIN_DRAG_PX or box[3] < MIN_DRAG_PX:
            return None  # klik tak sengaja
        self.boxes.append(box)
        return box

    def delete_at(self, point: tuple[int, int]) -> bool:
        """Hapus kotak terkecil yang memuat titik."""
        px, py = point
        inside = [b for b in self.boxes if b[0] <= px <= b[0] + b[2] and b[1] <= py <= b[1] + b[3]]
        if not inside:
            return False
        self.boxes.remove(min(inside, key=lambda b: b[2] * b[3]))
        return True


@dataclass(frozen=True)
class Item:
    """Satu citra yang dianotasi: kunci anotasi, berkas, dan jumlah wajah yang diharapkan (None = tidak diketahui)."""

    key: str
    path: Path
    expected: int | None


def fit_scale(width: int, height: int, limit: tuple[int, int] = FIT_WINDOW) -> float:
    """Skala tampilan supaya citra muat di jendela; tidak pernah memperbesar."""
    return min(1.0, limit[0] / width, limit[1] / height)


def _render(base: np.ndarray, item: Item, state: AnnotationState, view: Viewport,
            index: int, total: int, done: int) -> np.ndarray:
    """Gambar kotak di atas tampilan yang sudah diperkecil (koordinat jendela), bukan di citra asli —
    foto ponsel 20 MP terlalu berat untuk disalin setiap frame, dan garisnya tetap terlihat tebal."""
    shown = base.copy()
    for number, (x, y, w, h) in enumerate(sort_left_to_right(state.boxes), start=1):
        p1, p2 = view.to_window(x, y), view.to_window(x + w, y + h)
        cv2.rectangle(shown, p1, p2, (0, 255, 0), 2)
        cv2.putText(shown, str(number), (p1[0], max(p1[1] - 5, 44)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if state.drag_start is not None:
        cv2.rectangle(shown, view.to_window(*state.drag_start), view.to_window(*state.cursor), (0, 200, 255), 1)

    mismatch = item.expected is not None and len(state.boxes) != item.expected
    color = (60, 60, 255) if mismatch else (120, 255, 120)
    count = f"{len(state.boxes)}/{item.expected}" if item.expected is not None else str(len(state.boxes))
    header = f"[{index + 1}/{total}] {item.key} | kotak {count} | zoom {view.zoom}x | selesai {done}/{total}"
    cv2.rectangle(shown, (0, 0), (shown.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(shown, header, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    footer = "rambut->dagu, pipi->pipi, tanpa telinga | seret=kotak kanan=hapus n/p g z u c s q"
    cv2.rectangle(shown, (0, shown.shape[0] - 26), (shown.shape[1], shown.shape[0]), (20, 20, 20), -1)
    cv2.putText(shown, footer, (8, shown.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return shown


def dataset_items(paths: ProjectPaths, only_set: str | None) -> list[Item]:
    rows = sorted(read_metadata(paths.metadata), key=lambda r: r.file)
    if only_set:
        rows = [r for r in rows if r.set == only_set]
    return [Item(r.file, paths.raw / r.file, r.expected_faces) for r in rows if (paths.raw / r.file).exists()]


def folder_items(folder: Path) -> list[Item]:
    return [Item(key, folder / key, None) for key in list_folder_images(folder)]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--set", default=None, help="Hanya satu set (jarak/cahaya/multi/kosong/pose/ekspresi/oklusi)")
    parser.add_argument("--folder", default=None,
                        help="Anotasi foto lepas <folder>/S01/*.jpg (E9) alih-alih data/raw + metadata")
    parser.add_argument("--start", type=int, default=None, help="Mulai dari citra ke-N (1 = pertama)")
    parser.add_argument("--display-scale", type=float, default=1.0,
                        help="Skala jendela bila layar kecil, mis. 0.8 (mode --folder: dikalikan skala muat-layar)")


def run(args: argparse.Namespace, cfg: Config, paths: ProjectPaths | None = None) -> int:
    paths = paths or cfg.paths
    folder = Path(args.folder) if getattr(args, "folder", None) else None
    if folder is not None:
        if not folder.is_dir():
            print(f"[GAGAL] folder tidak ada: {folder}", file=sys.stderr)
            return 2
        items, store = folder_items(folder), folder_annotations_path(paths, folder)
    else:
        items, store = dataset_items(paths, args.set), paths.annotations
    if not items:
        print("Tidak ada foto untuk dianotasi. Rekam dulu: python -m pcdface capture ...", file=sys.stderr)
        return 1

    annotations = load_annotations(store)
    pending = [i for i, item in enumerate(items) if item.key not in annotations]
    if args.start is not None:
        index = min(max(args.start - 1, 0), len(items) - 1)
    else:
        index = pending[0] if pending else 0

    state = AnnotationState()
    view = Viewport(cfg.capture.width, cfg.capture.height, display_scale=args.display_scale)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)

    def commit() -> None:
        annotations[items[index].key] = list(state.boxes)

    try:
        while 0 <= index < len(items):
            item = items[index]
            image = cv2.imread(str(item.path))
            if image is None:
                print(f"[LEWAT] {item.key}: tidak terbaca")
                index += 1
                continue
            height, width = image.shape[:2]
            if folder is None and (width, height) != (view.width, view.height):
                print(f"[LEWAT] {item.key}: bukan {view.width}×{view.height}")
                index += 1
                continue
            scale = args.display_scale * (fit_scale(width, height) if folder is not None else 1.0)
            view = Viewport(width, height, display_scale=scale, center=(width / 2, height / 2))
            state.boxes = list(annotations.get(item.key, []))
            state.drag_start = None
            cached: tuple[tuple, np.ndarray] | None = None
            move = 0
            while move == 0:
                done = sum(1 for it in items if it.key in annotations)
                view_key = (view.zoom, view.origin)
                if cached is None or cached[0] != view_key:
                    cached = (view_key, view.render(image))
                cv2.imshow(WINDOW, _render(cached[1], item, state, view, index, len(items), done))
                key = cv2.waitKeyEx(20)
                if key == -1:
                    continue
                char = -1 if key in KEY_LEFT | KEY_RIGHT else lower_key(key)


                if char in (ord("n"), ord("d")) or key in KEY_RIGHT:
                    commit(); move = 1
                elif char in (ord("p"), ord("a")) or key in KEY_LEFT:
                    commit(); move = -1
                elif char == ord("g"):
                    commit()
                    remaining = [i for i, it in enumerate(items) if it.key not in annotations]
                    if remaining:
                        move = remaining[0] - index or 0
                        if move == 0:
                            print("  citra ini adalah yang pertama belum dianotasi")
                    else:
                        print("  semua citra sudah dianotasi")
                elif char == ord("z"):
                    view.cycle_zoom(state.cursor)
                elif char == ord("u") and state.boxes:
                    state.boxes.pop()
                elif char == ord("c"):
                    state.boxes.clear()
                elif char == ord("s"):
                    commit(); save_annotations(store, annotations)
                    print(f"  tersimpan ke {store}")
                elif char in (ord("q"), 27):
                    commit(); save_annotations(store, annotations)
                    print(f"\nTersimpan: {len(annotations)} citra -> {store}")
                    return 0
            save_annotations(store, annotations)
            index = max(index + move, 0)
    finally:
        cv2.destroyAllWindows()

    save_annotations(store, annotations)
    total = sum(len(v) for v in annotations.values())
    print(f"\nSelesai: {len(annotations)} citra, {total} kotak -> {store}")
    print("Berikutnya: python -m pcdface " + (f"eval-folder --input {folder}" if folder is not None else "validate"))
    return 0
