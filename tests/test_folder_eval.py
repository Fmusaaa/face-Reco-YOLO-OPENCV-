"""E9 / eval-folder: foto lepas per peserta, crop manual vs otomatis, LBPH 70/30 — pada data sintetis."""

from __future__ import annotations

import argparse
import dataclasses

import pandas as pd
import pytest

from pcdface.dataset.annotations import folder_annotations_path, list_folder_images, load_annotations
from pcdface.dataset.metadata import read_subjects
from pcdface.experiments import e9_folder
from pcdface.experiments.common import Record
from pcdface.synthetic import generate_folder
from pcdface.tools import eval_folder
from pcdface.tools.annotate import fit_scale


@pytest.fixture
def folder_setup(small_cfg, tmp_path):
    paths = small_cfg.paths.with_data_root(tmp_path / "data")
    paths = dataclasses.replace(paths, results=tmp_path / "results")
    folder = generate_folder(small_cfg, paths, tmp_path / "foto", photos_per_subject=6)
    return small_cfg, paths, folder


def test_generate_folder_and_annotation_helpers(folder_setup):
    cfg, paths, folder = folder_setup
    keys = list_folder_images(folder)
    assert len(keys) == 4 * 6 and keys[0] == "S01/001.jpg"
    store = folder_annotations_path(paths, folder)
    assert store.name == "folder_foto.json" and store.parent == paths.annotations.parent
    assert set(load_annotations(store)) == set(keys)
    assert fit_scale(4000, 3000) < 1 and fit_scale(800, 600) == 1.0


def test_load_folder_rejects_names_and_respects_consent(folder_setup):
    cfg, paths, folder = folder_setup
    annotations = load_annotations(folder_annotations_path(paths, folder))
    subjects = read_subjects(paths.subjects)
    del annotations["S01/001.jpg"]
    subjects["S02"] = dataclasses.replace(subjects["S02"], consent_research=False)
    report = e9_folder.load_folder(folder, annotations, subjects)
    assert report.unannotated == ["S01/001.jpg"]
    assert len(report.no_consent) == 6 and all(k.startswith("S02/") for k in report.no_consent)
    assert len(report.photos) == 24 - 1 - 6

    (folder / "Budi").mkdir()
    (folder / "Budi" / "x.jpg").write_bytes((folder / "S01" / "002.jpg").read_bytes())
    with pytest.raises(ValueError, match="kode peserta"):
        e9_folder.load_folder(folder, annotations, subjects)


def test_split_is_per_subject_seeded_and_consent_only(folder_setup):
    cfg, paths, folder = folder_setup
    report = e9_folder.load_folder(folder, load_annotations(folder_annotations_path(paths, folder)),
                                   read_subjects(paths.subjects))
    split = e9_folder.split_photos(report.photos, {"S01", "S02", "S03"}, 0.7, seed=42)
    assert split == e9_folder.split_photos(report.photos, {"S01", "S02", "S03"}, 0.7, seed=42)
    assert not any(k.startswith("S04/") for k in split)           # tanpa izin pengenalan
    for subject in ("S01", "S02", "S03"):
        parts = [v for k, v in split.items() if k.startswith(subject)]
        assert parts.count(e9_folder.TRAIN) == 4 and parts.count(e9_folder.TEST) == 2   # round(0,7 × 6)


def test_crop_agreement_geometry():
    record = Record(key="S01/a.jpg", group="S01", set="folder", gt=[(100, 100, 100, 100), (400, 100, 50, 50)],
                    boxes=[(110, 120, 100, 100)], scores=None, elapsed_ms=1.0, width=640, height=480)
    rows = e9_folder.crop_agreement([record], "haar")
    found, missed = rows
    assert found["ditemukan"] and missed["ditemukan"] is False and missed["iou"] == 0.0
    assert found["geser_x"] == pytest.approx(0.1) and found["geser_y"] == pytest.approx(0.2)
    assert (found["nx"], found["ny"], found["nw"], found["nh"]) == pytest.approx((0.1, 0.2, 1.0, 1.0))


def test_eval_folder_per_model_then_compare(folder_setup):
    cfg, paths, folder = folder_setup
    # model 1: OpenCV saja
    out = eval_folder.evaluate_folder(cfg, paths, folder, ["haar"], synthetic=True, save_crops=False, plots=False)
    detection = pd.read_csv(out / "e9_deteksi.csv")
    assert set(detection["detektor"]) == {"haar"} and not (out / "e9_selisih_detektor.csv").exists()
    # model 2: YOLO — perbandingan memuat keduanya, deteksi Haar dipakai ulang dari cache
    out = eval_folder.evaluate_folder(cfg, paths, folder, ["yolo_n"], synthetic=True, plots=True)
    detection = pd.read_csv(out / "e9_deteksi.csv")
    assert list(dict.fromkeys(detection["detektor"])) == ["haar", "yolo_n"]
    for column in ("precision", "recall", "f1", "ap"):
        valid = detection.dropna(subset=[f"{column}_low"])
        assert ((valid[f"{column}_low"] <= valid[column] + 1e-9) & (valid[column] <= valid[f"{column}_high"] + 1e-9)).all()
    assert set(pd.read_csv(out / "e9_selisih_detektor.csv")["metrik"]) == {"precision", "recall", "f1", "ap"}

    crops = pd.read_csv(out / "e9_crop.csv").set_index("detektor")
    assert (crops["wajah"] == 24).all() and (crops["iou_median"] > 0.5).all()
    recognition = pd.read_csv(out / "e9_pengenalan.csv").set_index("sumber")
    assert list(recognition.index) == ["manual", "haar", "yolo_n"]
    assert (recognition["foto_uji"] == 8).all() and recognition.loc["manual", "galeri"] == 16
    assert recognition.loc["manual", "akurasi"] > 0.5               # label tersambung: jauh di atas tebakan 1/4
    split = pd.read_csv(out / "e9_pembagian.csv")
    assert len(split) == 24 and set(split["bagian"]) == {"latih", "uji"}

    assert (out / "banding_crop" / "S01" / "001.jpg").exists() and (out / "crop" / "manual" / "S01").is_dir()
    assert len(list(out.glob("grafik1[6-9]_e9_*.png"))) == 4
    assert "BUKAN hasil penelitian" in (out / "RINGKASAN_E9.md").read_text(encoding="utf-8")

    # anotasi bertambah → cache lama tidak mencakup foto baru dan tidak diikutkan
    store = folder_annotations_path(paths, folder)
    (folder / "S01" / "099.jpg").write_bytes((folder / "S01" / "001.jpg").read_bytes())
    annotations = load_annotations(store)
    annotations["S01/099.jpg"] = annotations["S01/001.jpg"]
    from pcdface.dataset.annotations import save_annotations
    save_annotations(store, annotations)
    with pytest.raises(ValueError, match="belum ada deteksi"):
        eval_folder.evaluate_folder(cfg, paths, folder, [], compare_only=True, synthetic=True, plots=False)


def test_eval_folder_cli_arguments(cfg, tmp_path):
    def args(**kw):
        return argparse.Namespace(**{"input": None, "detectors": "haar", "compare_only": False, "no_crops": True,
                                     "no_plots": True, "synthetic": False, **kw})

    assert eval_folder.run(args(detectors="tidak_ada"), cfg) == 2
    assert eval_folder.run(args(), cfg) == 2                                     # --input wajib
    assert eval_folder.run(args(input=str(tmp_path / "tidak_ada")), cfg) == 2


def test_import_hand_crops_matches_by_content(folder_setup, tmp_path):
    import cv2
    import numpy as np

    from pcdface.evaluation.matching import iou
    from pcdface.tools import import_crops

    cfg, paths, folder = folder_setup
    store = folder_annotations_path(paths, folder)
    truth = load_annotations(store)
    store.unlink()                                             # hanya crop tangan yang jadi kotak manual
    crops = tmp_path / "crop-manual"
    keys = sorted(truth)
    for number, key in enumerate(keys, start=1):
        image = cv2.imread(str(folder / key))
        x, y, w, h = truth[key][0]
        crop = image[max(y - 8, 0):y + h + 8, max(x - 4, 0):x + w + 4]   # dipotong tangan: sedikit lebih longgar
        if number % 5 == 2:
            crop = cv2.resize(crop, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)   # disimpan diperkecil
        if number % 5 == 3:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)                              # tersimpan diputar
        # nama folder dan nomor berkas sengaja tidak sama dengan foto asal
        target = crops / key.split("/")[0] / f" {len(keys) - number + 1:03d}.JPG"
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), crop)
    first = cv2.imread(str(crops / "S01" / f" {len(keys):03d}.JPG"))
    cv2.imwrite(str(crops / "S01" / "duplikat.jpg"), first)                              # wajah yang sama dua kali
    noise = np.random.default_rng(1).integers(0, 255, (120, 90, 3), dtype=np.uint8)
    cv2.imwrite(str(crops / "S02" / "asing.png"), noise)                                 # bukan dari foto dataset

    from pcdface.dataset.annotations import save_annotations
    save_annotations(store, {"S04/001.jpg": [(1, 2, 3, 4)], "S04/999.jpg": [(5, 6, 7, 8)]})   # sisa impor lama
    report = import_crops.import_crops(cfg, paths, folder, crops)
    assert report.unmatched == ["S02/asing.png"] and report.duplicates == ["S01/duplikat.jpg"]
    assert report.dropped == ["S04/999.jpg"]                  # kotak basi tidak tertinggal
    imported = load_annotations(store)
    assert set(imported) == set(truth) and report.accepted_photos == len(truth)
    for key, boxes in imported.items():
        x, y, w, h = truth[key][0]
        expected = (max(x - 4, 0), max(y - 8, 0), w + 8, h + 16)
        assert len(boxes) == 1 and iou(boxes[0], expected) > 0.9, key
    assert (paths.results / import_crops.REPORT).exists()

    out = eval_folder.evaluate_folder(cfg, paths, folder, ["haar"], synthetic=True, save_crops=False, plots=False)
    assert pd.read_csv(out / "e9_deteksi.csv")["wajah"].iloc[0] == len(truth)


def test_default_crops_dir_accepts_hyphen(small_cfg, tmp_path):
    from pcdface.tools.import_crops import default_crops_dir

    paths = small_cfg.paths.with_data_root(tmp_path / "data")
    assert default_crops_dir(paths) == paths.manual_crops
    (tmp_path / "data" / "crop-manual").mkdir(parents=True)
    assert default_crops_dir(paths).name == "crop-manual"
    paths.manual_crops.mkdir()
    assert default_crops_dir(paths) == paths.manual_crops
