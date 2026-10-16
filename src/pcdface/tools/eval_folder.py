"""`python -m pcdface eval-folder` — E9 pada foto lepas per peserta (lihat `experiments/e9_folder.py`).

    python -m pcdface annotate --folder data/DATASET-MUSANG-PCD                      # crop manual dulu
    python -m pcdface eval-folder --input data/DATASET-MUSANG-PCD --detectors haar     # model OpenCV saja
    python -m pcdface eval-folder --input data/DATASET-MUSANG-PCD --detectors yolo_n   # lalu model YOLO
    python -m pcdface eval-folder --input data/DATASET-MUSANG-PCD                      # semua detektor config E9
    python -m pcdface eval-folder --input data/DATASET-MUSANG-PCD --compare-only       # tanpa deteksi ulang
    python -m pcdface eval-folder --synthetic                                          # uji jalur, tanpa model

Deteksi tiap detektor disimpan di `results/e9_deteksi/`; tabel di `results/e9/` selalu
membandingkan semua detektor yang deteksinya mencakup seluruh foto beranotasi.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

from pcdface.config import Config, ConfigError
from pcdface.dataset.annotations import folder_annotations_path, load_annotations
from pcdface.dataset.metadata import read_subjects
from pcdface.detection.registry import model_missing, supports_scores
from pcdface.experiments import e9_folder
from pcdface.experiments.runner import RunContext, write_snapshot
from pcdface.paths import ProjectPaths


def evaluate_folder(cfg: Config, paths: ProjectPaths, folder: Path, detectors: list[str], compare_only: bool = False,
                    synthetic: bool = False, save_crops: bool = True, plots: bool = True) -> Path:
    """Jalankan detektor terpilih (kecuali `compare_only`), lalu tulis perbandingan ke `results/e9/`."""
    subjects = read_subjects(paths.subjects)
    report = e9_folder.load_folder(folder, load_annotations(folder_annotations_path(paths, folder)), subjects)
    if report.unannotated:
        print(f"PERINGATAN: {len(report.unannotated)} foto belum dianotasi, tidak diikutkan "
              f"(python -m pcdface annotate --folder {folder})", file=sys.stderr)
    if report.no_consent:
        print(f"PERINGATAN: {len(report.no_consent)} foto dari peserta tanpa consent_research = ya di "
              f"{paths.subjects.name}, tidak diikutkan", file=sys.stderr)
    if not report.photos:
        raise ValueError(f"tidak ada foto beranotasi di {folder} — jalankan dulu: "
                         f"python -m pcdface annotate --folder {folder}")

    start = time.perf_counter()
    if not compare_only:
        for name in detectors:
            modes = e9_folder.MODES if supports_scores(name, cfg) else ("operating",)
            for mode in modes:
                print(f"== deteksi {name} ({mode}) pada {len(report.photos)} foto", flush=True)
                records = e9_folder.detect_photos(cfg, report.photos, name, mode, synthetic)
                e9_folder.save_records(e9_folder.cache_path(paths.results, name, mode), records)

    operating, ap = {}, {}
    known = list(cfg.raw.get("detectors", {}))
    cached = sorted({p.stem.rsplit("_", 1)[0] for p in (paths.results / e9_folder.CACHE_DIR).glob("*_operating.jsonl")},
                    key=lambda n: (known.index(n) if n in known else len(known), n))
    for name in cached:
        records = e9_folder.load_records(e9_folder.cache_path(paths.results, name, "operating"), report.photos)
        if records is None:
            print(f"PERINGATAN: deteksi {name} belum mencakup semua foto beranotasi — jalankan ulang "
                  f"--detectors {name}; tidak diikutkan", file=sys.stderr)
            continue
        operating[name] = records
        scored = e9_folder.load_records(e9_folder.cache_path(paths.results, name, "ap"), report.photos)
        if scored is not None:
            ap[name] = scored
    if not operating:
        raise ValueError("belum ada deteksi tersimpan — jalankan tanpa --compare-only")

    out_dir = paths.results / "e9"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    ctx = RunContext(cfg=cfg, paths=paths, name="e9", out_dir=out_dir, synthetic=synthetic)
    ctx.notes["folder"] = folder.name
    print(f"== perbandingan {', '.join(operating)} → {out_dir}", flush=True)
    try:
        e9_folder.run(ctx, report, operating, ap, subjects, save_crops=save_crops)
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    if plots:
        from pcdface.reporting.plots import plot_experiment

        figures = [p for p in plot_experiment("e9", out_dir) if p.suffix == ".png"]
        ctx.log(f"{len(figures)} grafik (PNG 300 dpi + PDF)")
    write_snapshot(ctx, time.perf_counter() - start)
    return out_dir


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default=None, help="Folder berisi subfolder per kode peserta (S01/, S02/, …)")
    parser.add_argument("--detectors", default=None,
                        help="Detektor yang dijalankan, dipisah koma (bawaan: experiments.e9.detectors)")
    parser.add_argument("--compare-only", action="store_true",
                        help="Jangan jalankan detektor; bandingkan deteksi yang sudah tersimpan")
    parser.add_argument("--no-crops", action="store_true", help="Jangan tulis crop dan gambar banding crop")
    parser.add_argument("--no-plots", action="store_true", help="Tabel saja, tanpa grafik")
    parser.add_argument("--synthetic", action="store_true",
                        help="Folder sintetis + FakeDetector (uji jalur; hasil ke results/synthetic/)")


def run(args: argparse.Namespace, cfg: Config) -> int:
    detectors = args.detectors.split(",") if args.detectors else list(cfg.experiments.e9.detectors)
    try:
        for name in detectors:
            cfg.detector(name)
    except ConfigError as error:
        print(f"[GAGAL] {error}", file=sys.stderr)
        return 2
    if not args.synthetic and not args.compare_only:
        missing = [n for n in detectors if model_missing(n, cfg)]
        if missing:
            print(f"[GAGAL] model {', '.join(missing)} belum ada — python -m pcdface download-models"
                  + (" / export-yolo" if "yolo_m" in missing else ""), file=sys.stderr)
            return 1

    with tempfile.TemporaryDirectory(prefix="pcdface_e9_") as temp:
        if args.synthetic:
            from pcdface.synthetic import generate_folder

            paths = cfg.paths.with_data_root(Path(temp) / "data")
            folder = generate_folder(cfg, paths, Path(temp) / "foto")
        else:
            if not args.input:
                print("[GAGAL] --input wajib, mis. --input data/DATASET-MUSANG-PCD", file=sys.stderr)
                return 2
            paths, folder = cfg.paths, Path(args.input)
            if not folder.is_dir():
                print(f"[GAGAL] folder tidak ada: {folder}", file=sys.stderr)
                return 2
        try:
            out_dir = evaluate_folder(cfg, paths, folder, detectors, args.compare_only, args.synthetic,
                                      save_crops=not args.no_crops, plots=not args.no_plots)
        except (ValueError, FileNotFoundError) as error:
            print(f"[GAGAL] {error}", file=sys.stderr)
            return 1
    print(f"\nHasil: {out_dir}\n  ringkasan: {out_dir / 'RINGKASAN_E9.md'}")
    if args.synthetic:
        print("Catatan: run sintetis hanya menguji jalur; angkanya BUKAN hasil penelitian.")
    else:
        print("Crop dan banding_crop berisi wajah: jangan dibagikan tanpa izin publikasi peserta.")
    return 0
