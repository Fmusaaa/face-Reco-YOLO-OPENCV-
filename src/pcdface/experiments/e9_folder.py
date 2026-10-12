"""E9 — Foto lepas per peserta: OpenCV vs YOLO, crop manual vs otomatis (PRD §9 E9).

Untuk foto yang dikumpulkan di luar protokol `capture` (mis. foto ponsel dari Drive) di
`<folder>/S01/*.jpg`, `<folder>/S02/*.jpg`, … — ukuran berapa pun, tanpa metadata:

    python -m pcdface annotate --folder data/DATASET-MUSANG-PCD          # 1. crop manual (ground truth)
    python -m pcdface eval-folder --input data/DATASET-MUSANG-PCD --detectors haar     # 2a. model OpenCV
    python -m pcdface eval-folder --input data/DATASET-MUSANG-PCD --detectors yolo_n   # 2b. model YOLO
    python -m pcdface eval-folder --input data/DATASET-MUSANG-PCD --compare-only       # 3. bandingkan lagi

Deteksi setiap detektor disimpan di `results/e9_deteksi/<detektor>_<mode>.jsonl`, jadi detektor
bisa dijalankan satu per satu; tabel perbandingan selalu memuat **semua** detektor yang sudah
punya deteksi untuk seluruh foto beranotasi.

Tiga bagian, semuanya terhadap kotak manual (PRD §7):

1. **Akurasi deteksi** — TP/FP/FN, precision, recall (Wilson 95%), F1 (bootstrap per peserta),
   AP (run ambang rendah terpisah, PRD §8.2), rerata IoU, FPPI, waktu; pada IoU 0,5 dan uji
   sensitivitas 0,3/0,4. Selisih antar detektor: bootstrap berpasangan per peserta.
2. **Crop manual vs crop otomatis** — setiap kotak manual dipasangkan satu-satu dengan deteksi
   ber-IoU tertinggi: IoU, geser pusat (proporsi lebar/tinggi kotak manual), rasio lebar dan
   tinggi, rasio aspek. Menunjukkan konvensi kotak tiap detektor (Haar persegi, YOLO gaya
   WIDER FACE) dan seberapa sering crop otomatis "cukup sama" dengan crop manual (IoU ≥ 0,5).
3. **Pengenalan LBPH dengan crop manual vs crop otomatis** — hanya peserta `consent_research`
   dan `consent_recognition` = ya, hanya foto berisi satu kotak manual. Foto tiap peserta dibagi
   acak `train_fraction` (galeri) / sisanya (uji) dengan `seed` config — pembagian yang **sama**
   untuk semua sumber kotak. Sumber detektor: kotak = deteksi ber-IoU tertinggi ≥ `iou_primary`
   terhadap kotak manual (aturan E8); foto uji tanpa deteksi itu dihitung gagal.

Keterbatasan yang wajib disebut: foto lepas tidak terkontrol (jarak, cahaya, kamera berbeda-beda),
jumlah peserta kecil sehingga interval bootstrap per peserta lebar, galeri dan foto uji LBPH
berasal dari kumpulan foto yang sama (akurasi bisa terlalu optimistis), dan waktu deteksi diukur
pada foto beresolusi berbeda-beda (bukan protokol E4).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import pandas as pd

from pcdface.boxes import Box, clip_box, sort_left_to_right
from pcdface.config import Config
from pcdface.dataset.annotations import list_folder_images
from pcdface.dataset.metadata import SUBJECT_ID, SubjectRow
from pcdface.detection.registry import build_detector
from pcdface.evaluation.matching import iou as box_iou
from pcdface.evaluation.matching import match_greedy
from pcdface.evaluation.stats import NAN_ESTIMATE, cluster_bootstrap, paired_difference, wilson
from pcdface.experiments.common import Record, ap_summary, operating_summary, paired, verdict
from pcdface.experiments.runner import RunContext
from pcdface.recognition.lbph import LBPHRecognizer, consenting_subjects, face_patch

MANUAL = "manual"
TRAIN, TEST = "latih", "uji"
CACHE_DIR = "e9_deteksi"
MODES = ("operating", "ap")
CROP_HEIGHT = 200            # tinggi tiap panel di gambar banding crop


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FolderPhoto:
    key: str                  # relatif terhadap folder, mis. "S01/001.jpg"
    subject: str
    path: Path
    gt: tuple[Box, ...]


@dataclass
class FolderReport:
    photos: list[FolderPhoto]
    unannotated: list[str]
    no_consent: list[str]


def load_folder(folder: Path, annotations: dict[str, list[Box]], subjects: dict[str, SubjectRow]) -> FolderReport:
    """Foto beranotasi dari peserta yang mengizinkan penelitian. Nama subfolder harus kode peserta."""
    report = FolderReport([], [], [])
    for key in list_folder_images(folder):
        subject = key.split("/", 1)[0]
        if not SUBJECT_ID.match(subject):
            raise ValueError(f"nama folder harus kode peserta seperti S01, bukan {subject!r} "
                             "(nama asli dilarang — CLAUDE.md aturan #5)")
        if key not in annotations:
            report.unannotated.append(key)
        elif subject not in subjects or not subjects[subject].consent_research:
            report.no_consent.append(key)
        else:
            report.photos.append(FolderPhoto(key, subject, folder / key, tuple(annotations[key])))
    return report


def read_image(photo: FolderPhoto) -> np.ndarray:
    image = cv2.imread(str(photo.path))
    if image is None:
        raise ValueError(f"foto tidak bisa dibaca: {photo.key}")
    return image


# ---------------------------------------------------------------------------
# Deteksi (disimpan per detektor supaya bisa dijalankan satu per satu)
# ---------------------------------------------------------------------------
def cache_path(results_root: Path, name: str, mode: str) -> Path:
    return results_root / CACHE_DIR / f"{name}_{mode}.jsonl"


def detect_photos(cfg: Config, photos: Sequence[FolderPhoto], name: str, mode: str = "operating",
                  synthetic: bool = False, log: Callable[[str], None] = print) -> list[Record]:
    """Jalankan satu detektor pada semua foto. Satu deteksi pemanasan dibuang (waktu inisialisasi)."""
    records = []
    with build_detector(name, cfg, mode=mode, synthetic=synthetic) as detector:
        for index, photo in enumerate(photos):
            image = read_image(photo)
            if index == 0:
                detector.detect(image)
            result = detector.detect(image)
            records.append(Record(
                key=photo.key, group=photo.subject, set="folder", gt=list(photo.gt), boxes=list(result.boxes),
                scores=list(result.scores) if result.scores is not None else None,
                elapsed_ms=result.elapsed_ms, width=image.shape[1], height=image.shape[0],
                labels={"detektor": name, "mode": mode},
            ))
            if (index + 1) % 20 == 0:
                log(f"{name} ({mode}): {index + 1}/{len(photos)} foto")
    return records


def save_records(path: Path, records: Sequence[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.to_json() + "\n" for r in records), encoding="utf-8")


def load_records(path: Path, photos: Sequence[FolderPhoto]) -> list[Record] | None:
    """Deteksi tersimpan untuk `photos`, dengan kotak manual terbaru. None bila ada foto yang belum dideteksi."""
    if not path.exists():
        return None
    stored = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        data = json.loads(line)
        data["gt"] = [tuple(b) for b in data["gt"]]
        data["boxes"] = [tuple(b) for b in data["boxes"]]
        data["positions_cm"] = tuple(data.get("positions_cm", ()))
        stored[data["key"]] = Record(**data)
    if any(photo.key not in stored for photo in photos):
        return None
    records = []
    for photo in photos:
        record = stored[photo.key]
        record.gt = list(photo.gt)       # anotasi bisa diperbaiki sesudah deteksi; deteksinya tetap sah
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# 1. Akurasi deteksi
# ---------------------------------------------------------------------------
def detection_rows(name: str, operating: Sequence[Record], ap: Sequence[Record] | None, cfg: Config,
                   device: str) -> list[dict[str, object]]:
    times = np.array([r.elapsed_ms for r in operating], dtype=float)
    megapixels = np.array([r.width * r.height / 1e6 for r in operating], dtype=float)
    rows = []
    for threshold in cfg.evaluation.iou_thresholds:
        row: dict[str, object] = {"detektor": name, "iou": threshold}
        row.update(operating_summary(operating, threshold, cfg).row())
        estimate = ap_summary(ap, threshold, cfg)[0] if ap else NAN_ESTIMATE
        row.update(estimate.as_dict("ap"))
        row.update({"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
                    "median_mp": float(np.median(megapixels)), "perangkat": device})
        rows.append(row)
    return rows


def pair_rows(operating: dict[str, list[Record]], ap: dict[str, list[Record]], cfg: Config) -> list[dict[str, object]]:
    """Selisih A − B pada foto yang sama, bootstrap berpasangan per peserta (PRD §12.2 butir 5)."""
    iou = cfg.evaluation.iou_primary
    rows = []
    for a, b in combinations(operating, 2):
        tests = [(m, operating[a], operating[b]) for m in ("precision", "recall", "f1")]
        if a in ap and b in ap:
            tests.append(("ap", ap[a], ap[b]))
        for metric, records_a, records_b in tests:
            estimate = paired(records_a, records_b, iou, cfg, metric)
            rows.append({"a": a, "b": b, "metrik": metric, **estimate.as_dict("selisih"),
                         "kesimpulan": verdict(estimate)})
    return rows


# ---------------------------------------------------------------------------
# 2. Crop manual vs crop otomatis
# ---------------------------------------------------------------------------
def crop_agreement(records: Sequence[Record], name: str) -> list[dict[str, object]]:
    """Satu baris per kotak manual: deteksi pasangannya (satu-satu, IoU tertinggi > 0) dan geometrinya.

    Koordinat ternormalisasi (`nx`, `ny`, `nw`, `nh`) = kotak deteksi dalam satuan kotak manual —
    kotak manual adalah persegi satuan [0, 1] × [0, 1] (grafik konvensi kotak)."""
    rows = []
    for record in records:
        ordered = sort_left_to_right(record.gt)
        match = match_greedy(ordered, record.boxes, 1e-9)
        partner = {gt_index: (pred_index, value) for gt_index, pred_index, value in match.pairs}
        for index, (gx, gy, gw, gh) in enumerate(ordered):
            row: dict[str, object] = {"detektor": name, "foto": record.key, "peserta": record.group,
                                      "wajah": index + 1, "lebar_manual_px": gw, "tinggi_manual_px": gh,
                                      "aspek_manual": gw / gh}
            if index in partner:
                px, py, pw, ph = record.boxes[partner[index][0]]
                row.update({
                    "ditemukan": True, "iou": partner[index][1],
                    "geser_x": (px + pw / 2 - gx - gw / 2) / gw, "geser_y": (py + ph / 2 - gy - gh / 2) / gh,
                    "rasio_lebar": pw / gw, "rasio_tinggi": ph / gh, "aspek_deteksi": pw / ph,
                    "nx": (px - gx) / gw, "ny": (py - gy) / gh, "nw": pw / gw, "nh": ph / gh,
                })
            else:
                row.update({"ditemukan": False, "iou": 0.0})
            rows.append(row)
    return rows


def crop_summary(faces: pd.DataFrame, iou_primary: float, level: float) -> pd.DataFrame:
    rows = []
    for name, part in faces.groupby("detektor", sort=False):
        found = part[part["ditemukan"]]
        good = int((part["iou"] >= iou_primary).sum())
        row: dict[str, object] = {"detektor": name, "wajah": len(part), "ditemukan": len(found)}
        row.update(wilson(good, len(part), level).as_dict("crop_iou_cukup"))
        quartiles = found["iou"].quantile([0.25, 0.5, 0.75]) if len(found) else pd.Series([math.nan] * 3)
        row.update({
            "iou_median": float(quartiles.iloc[1]), "iou_q1": float(quartiles.iloc[0]), "iou_q3": float(quartiles.iloc[2]),
            "geser_x_median": float(found["geser_x"].abs().median()) if len(found) else math.nan,
            "geser_y_median": float(found["geser_y"].abs().median()) if len(found) else math.nan,
            "rasio_lebar_median": float(found["rasio_lebar"].median()) if len(found) else math.nan,
            "rasio_tinggi_median": float(found["rasio_tinggi"].median()) if len(found) else math.nan,
            "aspek_deteksi_median": float(found["aspek_deteksi"].median()) if len(found) else math.nan,
            "aspek_manual_median": float(part["aspek_manual"].median()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _crop(image: np.ndarray, box: Box) -> np.ndarray | None:
    clipped = clip_box(tuple(int(v) for v in box), image.shape[1], image.shape[0])  # type: ignore[arg-type]
    if clipped is None:
        return None
    x, y, w, h = clipped
    return image[y:y + h, x:x + w]


def _panel(crop: np.ndarray | None, title: str, subtitle: str) -> np.ndarray:
    if crop is None:
        panel = np.full((CROP_HEIGHT, CROP_HEIGHT, 3), 60, np.uint8)
        cv2.putText(panel, "tidak", (60, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(panel, "terdeteksi", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
    else:
        width = max(1, int(round(crop.shape[1] * CROP_HEIGHT / crop.shape[0])))
        interpolation = cv2.INTER_AREA if crop.shape[0] > CROP_HEIGHT else cv2.INTER_LINEAR
        panel = cv2.resize(crop, (width, CROP_HEIGHT), interpolation=interpolation)
    header = np.full((44, panel.shape[1], 3), 245, np.uint8)
    cv2.putText(header, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(header, subtitle, (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    return np.vstack([header, panel])


def _hstack(panels: list[np.ndarray], gap: int = 8) -> np.ndarray:
    height = max(p.shape[0] for p in panels)
    spacer = np.full((height, gap, 3), 255, np.uint8)
    parts = []
    for panel in panels:
        pad = np.full((height - panel.shape[0], panel.shape[1], 3), 255, np.uint8)
        parts += [np.vstack([panel, pad]), spacer]
    return np.hstack(parts[:-1])


def write_crops(photos: Sequence[FolderPhoto], operating: dict[str, list[Record]], out_dir: Path) -> int:
    """Crop manual + crop tiap detektor per wajah, dan gambar banding berdampingan per foto.

    `crop/<sumber>/<S01>/<foto>_f<N>.jpg` (N kiri → kanan; deteksi palsu: `_palsu<N>`),
    `banding_crop/<S01>/<foto>.jpg` (baris = wajah, kolom = manual | detektor …).
    Berisi wajah: hanya untuk pemeriksaan internal (results/ di-gitignore)."""
    by_key = {name: {r.key: r for r in records} for name, records in operating.items()}
    written = 0
    for photo in photos:
        image = read_image(photo)
        stem = Path(photo.key).stem
        ordered = sort_left_to_right(photo.gt)
        rows = []
        for index, gt in enumerate(ordered, start=1):
            manual = _crop(image, gt)
            panels = [_panel(manual, "manual", f"{gt[2]}x{gt[3]} px")]
            if manual is not None:
                target = out_dir / "crop" / MANUAL / photo.subject / f"{stem}_f{index}.jpg"
                target.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target), manual)
            rows.append(panels)
        for name, records in by_key.items():
            record = records[photo.key]
            match = match_greedy(ordered, record.boxes, 1e-9)
            partner = {gt_index: (pred_index, value) for gt_index, pred_index, value in match.pairs}
            for index in range(len(ordered)):
                crop, subtitle = None, ""
                if index in partner:
                    box = record.boxes[partner[index][0]]
                    crop, subtitle = _crop(image, box), f"IoU {partner[index][1]:.2f}".replace(".", ",")
                    if crop is not None:
                        target = out_dir / "crop" / name / photo.subject / f"{stem}_f{index + 1}.jpg"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(target), crop)
                rows[index].append(_panel(crop, name, subtitle))
            for number, j in enumerate((j for j, used in enumerate(match.pred_matched) if not used), start=1):
                crop = _crop(image, record.boxes[j])
                if crop is not None:
                    target = out_dir / "crop" / name / photo.subject / f"{stem}_palsu{number}.jpg"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(target), crop)
        if rows:
            lines = [_hstack(panels) for panels in rows]
            width = max(line.shape[1] for line in lines)
            sheet = np.vstack([np.hstack([line, np.full((line.shape[0], width - line.shape[1], 3), 255, np.uint8)])
                               for line in lines])
            target = out_dir / "banding_crop" / photo.subject / f"{stem}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), sheet)
            written += 1
    return written


# ---------------------------------------------------------------------------
# 3. Pengenalan LBPH: crop manual vs crop otomatis
# ---------------------------------------------------------------------------
def split_photos(photos: Sequence[FolderPhoto], allowed: set[str], fraction: float, seed: int) -> dict[str, str]:
    """Bagi foto satu-wajah tiap peserta berizin menjadi latih/uji. Peserta dengan < 2 foto tidak diikutkan."""
    by_subject: dict[str, list[str]] = defaultdict(list)
    for photo in photos:
        if photo.subject in allowed and len(photo.gt) == 1:
            by_subject[photo.subject].append(photo.key)
    rng = np.random.default_rng(seed)
    split: dict[str, str] = {}
    for subject in sorted(by_subject):
        keys = sorted(by_subject[subject])
        if len(keys) < 2:
            continue
        n_train = min(max(1, int(math.floor(fraction * len(keys) + 0.5))), len(keys) - 1)
        for rank, i in enumerate(rng.permutation(len(keys))):
            split[keys[i]] = TRAIN if rank < n_train else TEST
    return split


def best_match(gt: Box, boxes: Sequence[Box], iou_threshold: float) -> Box | None:
    """Deteksi dengan IoU tertinggi terhadap kotak manual, bila ≥ ambang (aturan E8)."""
    scored = [(box_iou(gt, box), box) for box in boxes]
    scored = [item for item in scored if item[0] >= iou_threshold]
    return max(scored)[1] if scored else None


@dataclass
class RecProbe:
    key: str
    subject: str
    source: str
    detected: bool
    nearest: str = ""
    distance: float = math.nan

    @property
    def correct(self) -> bool:
        return self.detected and self.nearest == self.subject

    def identified(self, threshold: float) -> bool:
        return self.correct and self.distance <= threshold


@dataclass
class RecognitionResult:
    probes: list[RecProbe]
    gallery: dict[str, int]          # wajah galeri per sumber
    gallery_missed: dict[str, int]   # foto latih yang wajahnya tidak terdeteksi per sumber


def recognize(cfg: Config, photos: Sequence[FolderPhoto], split: dict[str, str],
              operating: dict[str, list[Record]]) -> RecognitionResult:
    """Latih LBPH per sumber kotak pada foto latih, uji pada foto uji (pembagian yang sama)."""
    iou = cfg.evaluation.iou_primary
    by_key = {name: {r.key: r for r in records} for name, records in operating.items()}
    sources = [MANUAL, *operating]
    patches: dict[str, dict[str, np.ndarray | None]] = {s: {} for s in sources}
    chosen = [p for p in photos if p.key in split]
    for photo in chosen:
        image = read_image(photo)
        gt = photo.gt[0]
        patches[MANUAL][photo.key] = face_patch(image, gt, cfg.recognition.face_size)
        for name in operating:
            box = best_match(gt, by_key[name][photo.key].boxes, iou)
            patches[name][photo.key] = None if box is None else face_patch(image, box, cfg.recognition.face_size)

    probes: list[RecProbe] = []
    gallery, missed = {}, {}
    for source in sources:
        train = [(patches[source][p.key], p.subject) for p in chosen if split[p.key] == TRAIN]
        usable = [(patch, label) for patch, label in train if patch is not None]
        gallery[source], missed[source] = len(usable), len(train) - len(usable)
        recognizer = None
        if usable:
            recognizer = LBPHRecognizer(cfg.recognition)
            recognizer.train([patch for patch, _ in usable], [label for _, label in usable])
        for photo in chosen:
            if split[photo.key] != TEST:
                continue
            patch = patches[source][photo.key]
            probe = RecProbe(photo.key, photo.subject, source, detected=patch is not None and recognizer is not None)
            if probe.detected:
                prediction = recognizer.predict(patch)  # type: ignore[union-attr]
                probe.nearest, probe.distance = prediction.nearest, prediction.distance
            probes.append(probe)
    return RecognitionResult(probes, gallery, missed)


def _counts(probes: Sequence[RecProbe], hit: Callable[[RecProbe], bool]) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for probe in probes:
        counts[probe.subject][0] += int(hit(probe))
        counts[probe.subject][1] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def _pooled(counts: dict[str, tuple[int, int]]) -> Callable[[Sequence[str]], float]:
    def statistic(keys: Sequence[str]) -> float:
        k = sum(counts[key][0] for key in keys if key in counts)
        n = sum(counts[key][1] for key in keys if key in counts)
        return k / n if n else math.nan

    return statistic


def recognition_rows(result: RecognitionResult, cfg: Config) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """(ringkasan per sumber, selisih akurasi manual − detektor per peserta berpasangan)."""
    level, threshold = cfg.stats.ci_level, cfg.recognition.max_distance
    by_source: dict[str, list[RecProbe]] = defaultdict(list)
    for probe in result.probes:
        by_source[probe.source].append(probe)
    rows = []
    for source in result.gallery:
        probes = by_source.get(source, [])
        n = len(probes)
        counts = _counts(probes, lambda p: p.correct)
        boot = cluster_bootstrap(sorted(counts), _pooled(counts), cfg.stats.bootstrap_resamples, level, cfg.seed) \
            if counts else NAN_ESTIMATE
        distances = [p.distance for p in probes if p.detected]
        row: dict[str, object] = {"sumber": source, "peserta": len(counts), "galeri": result.gallery[source],
                                  "galeri_tak_terdeteksi": result.gallery_missed[source], "foto_uji": n,
                                  "tak_terdeteksi": sum(not p.detected for p in probes)}
        row.update(wilson(sum(p.correct for p in probes), n, level).as_dict("akurasi"))
        row.update({"akurasi_boot_low": boot.low, "akurasi_boot_high": boot.high})
        row.update(wilson(sum(p.identified(threshold) for p in probes), n, level).as_dict("dir"))
        row["rerata_jarak_lbph"] = float(np.mean(distances)) if distances else math.nan
        rows.append(row)

    diffs = []
    manual = _counts(by_source.get(MANUAL, []), lambda p: p.correct)
    for source in result.gallery:
        if source == MANUAL:
            continue
        other = _counts(by_source.get(source, []), lambda p: p.correct)
        groups = sorted(set(manual) & set(other))
        estimate = paired_difference(groups, _pooled(manual), _pooled(other), cfg.stats.bootstrap_resamples,
                                     level, cfg.seed) if groups else NAN_ESTIMATE
        diffs.append({"a": MANUAL, "b": source, "metrik": "akurasi", **estimate.as_dict("selisih"),
                      "kesimpulan": verdict(estimate)})
    return rows, diffs


# ---------------------------------------------------------------------------
# Ringkasan naratif
# ---------------------------------------------------------------------------
def _num(value: float, decimals: int = 3) -> str:
    return "–" if value is None or (isinstance(value, float) and math.isnan(value)) \
        else f"{value:.{decimals}f}".replace(".", ",")


def _ci(row: pd.Series, column: str) -> str:
    low, high = row.get(f"{column}_low", math.nan), row.get(f"{column}_high", math.nan)
    if pd.isna(low) or pd.isna(high):
        return _num(float(row[column]))
    return f"{_num(float(row[column]))} [{_num(float(low))}; {_num(float(high))}]"


def overview(detection: pd.DataFrame, pairs: pd.DataFrame, crops: pd.DataFrame, recognition: pd.DataFrame,
             rec_pairs: pd.DataFrame, cfg: Config, n_photos: int, n_subjects: int, synthetic: bool) -> str:
    iou = cfg.evaluation.iou_primary
    lines = ["# E9 — Foto lepas per peserta: ringkasan", ""]
    if synthetic:
        lines += ["> **Data sintetis + FakeDetector — angka BUKAN hasil penelitian.**", ""]
    lines += [f"{n_photos} foto beranotasi dari {n_subjects} peserta. Kotak manual = ground truth. "
              "Interval: Wilson 95% untuk proporsi, bootstrap per peserta untuk F1/AP. "
              "Dua detektor hanya disebut berbeda bila interval selisihnya tidak memuat 0.", "",
              f"## 1. Akurasi deteksi (IoU ≥ {_num(iou, 1)})", ""]
    main = detection[np.isclose(detection["iou"], iou)]
    for _, row in main.iterrows():
        lines.append(f"- **{row['detektor']}**: precision {_ci(row, 'precision')}, recall {_ci(row, 'recall')}, "
                     f"F1 {_ci(row, 'f1')}, AP {_ci(row, 'ap')}; TP {row['TP']}, FP {row['FP']}, FN {row['FN']}; "
                     f"rerata IoU {_num(float(row['rerata_iou']))}; median {_num(float(row['median_ms']), 1)} ms/foto.")
    for _, row in pairs.iterrows():
        lines.append(f"- Selisih {row['metrik']} {row['a']} − {row['b']}: {_ci(row, 'selisih')} → {row['kesimpulan']}.")
    lines += ["", "## 2. Crop manual vs crop otomatis", ""]
    for _, row in crops.iterrows():
        lines.append(f"- **{row['detektor']}**: {row['ditemukan']}/{row['wajah']} wajah punya crop otomatis; "
                     f"crop dengan IoU ≥ {_num(iou, 1)} terhadap crop manual {_ci(row, 'crop_iou_cukup')}; "
                     f"median IoU {_num(float(row['iou_median']))} (Q1–Q3 {_num(float(row['iou_q1']))}–"
                     f"{_num(float(row['iou_q3']))}); lebar crop otomatis {_num(float(row['rasio_lebar_median']), 2)}× "
                     f"dan tinggi {_num(float(row['rasio_tinggi_median']), 2)}× crop manual; aspek lebar/tinggi "
                     f"{_num(float(row['aspek_deteksi_median']), 2)} vs manual {_num(float(row['aspek_manual_median']), 2)}.")
    lines += ["", "## 3. Pengenalan LBPH: crop manual vs crop otomatis", ""]
    if recognition.empty:
        lines.append("- Tidak dijalankan: butuh ≥ 2 foto satu-wajah per peserta dengan izin pengenalan.")
    for _, row in recognition.iterrows():
        lines.append(f"- **{row['sumber']}**: galeri {row['galeri']} wajah, {row['foto_uji']} foto uji "
                     f"({row['tak_terdeteksi']} tak terdeteksi); akurasi rank-1 {_ci(row, 'akurasi')}; "
                     f"DIR (jarak ≤ {_num(cfg.recognition.max_distance, 0)}) {_ci(row, 'dir')}.")
    for _, row in rec_pairs.iterrows():
        lines.append(f"- Selisih akurasi {row['a']} − {row['b']}: {_ci(row, 'selisih')} → {row['kesimpulan']}.")
    lines += ["", "## Keterbatasan", "",
              "- Foto lepas tidak terkontrol: jarak, cahaya, kamera, dan resolusi berbeda-beda per foto.",
              f"- Hanya {n_subjects} peserta: interval bootstrap per peserta lebar; klaim dibuat hati-hati.",
              "- Galeri dan foto uji LBPH berasal dari kumpulan foto yang sama — akurasi bisa terlalu optimistis.",
              "- Waktu deteksi diukur pada foto beresolusi berbeda-beda; perbandingan kecepatan resmi ada di E4.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Jalankan
# ---------------------------------------------------------------------------
def run(ctx: RunContext, report: FolderReport, operating: dict[str, list[Record]], ap: dict[str, list[Record]],
        subjects: dict[str, SubjectRow], save_crops: bool = True) -> None:
    """Tulis semua tabel E9 dari deteksi yang sudah ada. `operating`/`ap`: {detektor: catatan}."""
    cfg, photos = ctx.cfg, report.photos
    ctx.notes["sampel"] = {"foto": len(photos), "wajah_manual": sum(len(p.gt) for p in photos),
                           "peserta": len({p.subject for p in photos}),
                           "belum_dianotasi": len(report.unannotated), "tanpa_izin_penelitian": len(report.no_consent)}
    ctx.notes["detektor"] = list(operating)

    rows = []
    for name, records in operating.items():
        rows += detection_rows(name, records, ap.get(name), cfg, ctx.device(name))
    detection = pd.DataFrame(rows)
    ctx.table(detection, "e9_deteksi", "E9 — Akurasi deteksi terhadap crop manual (P/R Wilson 95%, F1/AP bootstrap per peserta)",
              "IoU 0,5 = ambang utama; 0,3 dan 0,4 = uji sensitivitas konvensi kotak (PRD §7). "
              "Waktu = median per foto pada resolusi asli foto (bukan protokol E4).")
    pairs = pd.DataFrame(pair_rows(operating, ap, cfg))
    if not pairs.empty:
        ctx.table(pairs, "e9_selisih_detektor", "E9 — Selisih metrik A − B pada foto yang sama, IoU 0,5 "
                  "(bootstrap berpasangan per peserta)")

    faces = pd.DataFrame([row for name, records in operating.items() for row in crop_agreement(records, name)])
    crops = crop_summary(faces, cfg.evaluation.iou_primary, cfg.stats.ci_level) if not faces.empty else pd.DataFrame()
    if not faces.empty:
        ctx.table(crops, "e9_crop", "E9 — Crop otomatis dibanding crop manual (per wajah, pasangan IoU tertinggi)",
                  "crop_iou_cukup = proporsi wajah yang crop otomatisnya ber-IoU ≥ 0,5 dengan crop manual (Wilson 95%). "
                  "geser = |selisih pusat| dalam proporsi lebar/tinggi crop manual; rasio = ukuran crop otomatis ÷ manual.")
        ctx.table(faces, "e9_crop_per_wajah", "E9 — Geometri crop per wajah (audit)")

    allowed = consenting_subjects(subjects)
    split = split_photos(photos, allowed, cfg.experiments.e9.train_fraction, cfg.seed)
    recognition, rec_pairs = pd.DataFrame(), pd.DataFrame()
    if split and {s for k, s in split.items() if s == TEST}:
        result = recognize(cfg, photos, split, operating)
        rec_rows, diff_rows = recognition_rows(result, cfg)
        recognition, rec_pairs = pd.DataFrame(rec_rows), pd.DataFrame(diff_rows)
        ctx.table(recognition, "e9_pengenalan", "E9 — Akurasi pengenalan LBPH: crop manual vs crop otomatis",
                  f"Pembagian {cfg.experiments.e9.train_fraction:.0%} latih / sisanya uji per peserta, seed {cfg.seed}, "
                  "sama untuk semua sumber. akurasi = rank-1 (Wilson 95%; akurasi_boot = bootstrap per peserta). "
                  f"DIR = benar dan jarak LBPH ≤ {cfg.recognition.max_distance:g}. Foto uji tanpa deteksi = gagal.")
        if not rec_pairs.empty:
            ctx.table(rec_pairs, "e9_pengenalan_selisih", "E9 — Selisih akurasi pengenalan manual − detektor "
                      "(bootstrap berpasangan per peserta)")
        ctx.table(pd.DataFrame([{"foto": p.key, "peserta": p.subject, "sumber": p.source, "terdeteksi": p.detected,
                                 "prediksi": p.nearest, "jarak_lbph": p.distance, "benar": p.correct}
                                for p in result.probes]), "e9_prediksi", "E9 — Prediksi per foto uji (audit)")
        ctx.table(pd.DataFrame([{"foto": k, "peserta": k.split("/", 1)[0], "bagian": v} for k, v in sorted(split.items())]),
                  "e9_pembagian", "E9 — Pembagian latih/uji LBPH")
    else:
        ctx.log("pengenalan dilewati: butuh ≥ 2 foto satu-wajah per peserta dengan consent_recognition = ya")

    if save_crops:
        written = write_crops(photos, operating, ctx.out_dir)
        ctx.log(f"{written} gambar banding crop → {ctx.out_dir / 'banding_crop'} (berisi wajah: internal saja)")

    text = overview(detection, pairs, crops, recognition, rec_pairs, cfg, len(photos),
                    len({p.subject for p in photos}), ctx.synthetic)
    (ctx.out_dir / "RINGKASAN_E9.md").write_text(text, encoding="utf-8")
