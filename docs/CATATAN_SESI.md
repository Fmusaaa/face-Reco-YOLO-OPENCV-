# Catatan Serah Terima Sesi (7 Oktober 2026)

Ringkasan sesi Claude Code cloud untuk dilanjutkan di Claude Code lokal. Isinya tanya-jawab
sejauh ini, tugas yang belum selesai, dan saran paper. Spesifikasi lengkap ada di `docs/PRD.md` (v4)
dan aturan permanen di `CLAUDE.md`.

> Untuk Claude Code lokal: baca `CLAUDE.md`, lalu berkas ini. Kode Fase 6 sudah selesai di branch
> `claude/face-recognition-yolo-crop-mmyiyt` (PR Fmusaaa/face-detection#1, masih draf).

---

## 1. Yang sudah dibangun (Fase 6, commit `ad6cd0b`)

- **Tiga detektor dibandingkan**, semua *pretrained* dan tidak di-*fine-tune* (PRD §4):
  Haar Cascade (OpenCV), MediaPipe BlazeFace short/full (Tasks API), dan YOLOv8-face lewat OpenCV DNN
  (`yolo_n` P0; `yolo_m` P1 lewat `export-yolo`). YOLO ikut di E1–E4, E6, dan E7.
- **`crop-faces`** memotong setiap wajah per foto (urut kiri → kanan) ke `results/crop_wajah/`
  dan menulis manifest `crop_wajah.csv`.
- **Pengenalan LBPH berizin** (`enroll`, `crop-faces --recognize`, tombol `r` di demo). Label hanya
  `S01`, `S02`, …, dan wajah yang tidak cukup mirip diberi label `unknown` (jarak > 65).
- **E8** mengukur akurasi pengenalan per kondisi, dengan grafik 13–15.
- Verifikasi di cloud: `pytest` 149 lulus, `pytest -m models` 12 lulus, `selftest` 7/7,
  `run all --synthetic` + `report` OK.

## 2. Ringkasan tanya-jawab

| Pertanyaan | Jawaban singkat |
|---|---|
| `ConfigError: 'yolo_n' tidak ada` | Masih di `main`. Pindah ke branch `claude/face-recognition-yolo-crop-mmyiyt` (atau tunggu PR di-merge). |
| `model yolov8n-face.onnx tidak ada` | Jalankan `python -m pcdface download-models`. |
| Foto dari Drive ditaruh di mana? | `data/DATASET-MUSANG-PCD/S01/*.jpg`, `S02/…` (nama folder harus kode peserta, bukan nama asli). |
| `subjects.csv` di mana? | `data/subjects.csv`, kolom `subject_id,consent_research,consent_publication,consent_date,consent_recognition`, isi `ya`/`tidak`. |
| Cara melatih pengenalan | `python -m pcdface enroll --folder data/DATASET-MUSANG-PCD` → model ke `data/recognition/`. |
| Crop otomatis | `python -m pcdface crop-faces --input data/DATASET-MUSANG-PCD` (tambah `--recognize` untuk mengelompokkan per S01/unknown). |
| Perlu crop grayscale? | Tidak wajib. LBPH sudah mengubah ke abu-abu 200×200 di dalam. Crop berwarna lebih berguna untuk laporan. |
| Cara cek akurasi | Bagi foto 70/30 (`data/latih`, `data/uji`), `enroll` pada latih, `crop-faces --recognize` pada uji, lalu hitung benar/total + interval Wilson. Jangan menguji dengan foto yang dipakai `enroll`. |
| Model kita OpenCV + YOLO? | Ya. YOLO dijalankan lewat OpenCV DNN (`cv2.dnn.readNetFromONNX`), tanpa `ultralytics`/PyTorch. |
| Hasil disimpan di mana? | Crop: `results/crop_wajah/`. Tangkapan demo (tombol `s`): `results/demo/`. Model LBPH: `data/recognition/`. Tabel eksperimen: `results/e1…e8/`. Ringkasan: `results/RINGKASAN.md`. |
| Ini perbandingan YOLO vs MediaPipe vs Haar? | Ya, itu inti penelitian (RQ1–RQ7). LBPH adalah tambahan (RQ8). |
| MediaPipe belum ditraining? | Benar, dan memang tidak perlu. Ketiga detektor *pretrained*. Yang dilatih hanya LBPH lewat `enroll`. |
| Yang mengenali orang cuma LBPH? | Ya. YOLO hanya mencari letak wajah. Alurnya: YOLO → crop → abu-abu 200×200 → LBPH → `S01` (jarak ≤ 65) atau `unknown`. |
| Persen di kotak YOLO itu akurasi? | **Bukan.** `19.1%` = lebar wajah ÷ lebar frame; `s=0.87` = skor keyakinan YOLO. Akurasi asli = F1/AP (E1–E7) dan rank-1/DIR/FAR (E8), yang dihitung terhadap kunci jawaban. |
| Push ke GitHub selain data | Dari Mac: `git add -A`, cek `git status` (pastikan tidak ada `data/`, `results/`, `.onnx`, `.tflite`, `.pt`), lalu `git commit` dan `git push`. `.gitignore` sudah menyaring semuanya. |

## 3. Tugas yang belum selesai

Urutan sesuai jadwal PRD §15 (minggu 3 = 7–13 Oktober).

### Wajib
1. **Data dan izin**
   - [ ] Formulir persetujuan v4 (bagian 7: izin pengenalan) ditandatangani semua peserta.
   - [ ] Isi `data/subjects.csv`, termasuk kolom `consent_recognition`.
   - [ ] Lengkapi pengambilan data per set (jarak, cahaya, multi, pose, ekspresi, oklusi) dengan `capture`.
2. **Anotasi**
   - [ ] `python -m pcdface annotate` (bisa dicicil per set: `--set pose`).
   - [ ] `python -m pcdface validate` sampai bersih dari GALAT.
3. **Eksperimen**
   - [ ] `python -m pcdface enroll` (galeri dari foto kondisi acuan dataset).
   - [ ] `python -m pcdface run all`, lalu `python -m pcdface report`.
   - [ ] Baca `results/RINGKASAN.md` dan grafik 1–15.
4. **Akurasi pengenalan pada foto Drive**
   - [ ] Split 70/30 → `enroll` latih → `crop-faces --recognize` uji → hitung akurasi + Wilson.
5. **GitHub**
   - [ ] Review dan merge PR Fmusaaa/face-detection#1 ke `main`.
   - [ ] (Opsional) pasang Claude GitHub App supaya Claude bisa memantau PR:
         https://github.com/apps/claude/installations/select_target

### Opsional (sudah ditawarkan, belum dikerjakan)
- [ ] `yolo_m` (yolov8m-face, seperti repo referensi): `python -m pcdface export-yolo`, lalu tambahkan
      `yolo_m` ke daftar detektor di `configs/experiment.yaml` **sebelum** melihat hasil (aturan #6).
- [ ] Opsi `--gray` untuk `crop-faces`.
- [ ] Perintah resmi `eval-folder` (otomatisasi split 70/30 + akurasi + Wilson) untuk menggantikan skrip manual.
- [ ] Uji jalur CPU MediaPipe di laptop Windows/Linux anggota lain (`pytest`, `run all --synthetic`).

### Jangan lupa (aturan keras)
- Jangan commit apa pun dari `data/` (foto, `subjects.csv`, model LBPH), citra `results/`, `.onnx`, `.tflite`, `.pt`.
- Jangan tulis nama asli peserta di mana pun, termasuk nama folder dan paper.
- Jangan pasang `ultralytics`/PyTorch/`opencv-python` di `.venv`.
- Jangan ubah keputusan PRD §12.2 setelah melihat hasil.

## 4. Saran paper

### Judul (pilih satu)
- *Perbandingan Haar Cascade, MediaPipe BlazeFace, dan YOLOv8-Face untuk Deteksi Wajah pada Variasi Jarak,
  Pencahayaan, dan Pose, dengan Pengenalan Identitas LBPH*
- *Evaluasi Detektor Wajah Klasik dan Berbasis Deep Learning pada Kondisi Akuisisi Nyata*

### Struktur
1. **Pendahuluan**: masalah (deteksi wajah di kondisi nyata), celah (perbandingan klasik vs modern pada
   jarak dan ukuran wajah terukur), RQ1–RQ8, kontribusi.
2. **Tinjauan pustaka**: Viola-Jones/Haar, BlazeFace, YOLO untuk wajah (YOLOv8-face, WIDER FACE), LBP/LBPH, CLAHE.
3. **Metode**: protokol akuisisi (jarak terukur, cahaya, pose, ekspresi, oklusi), anotasi manual, tiga
   detektor beserta parameter, pencocokan IoU, metrik, LBPH + ambang, statistik (Wilson, bootstrap per subjek).
4. **Hasil**: per RQ, satu grafik utama + satu tabel per RQ.
5. **Pembahasan**: kenapa tiap detektor gagal di kondisi tertentu (ukuran minimum, pose profil, redup), trade-off kecepatan vs akurasi.
6. **Keterbatasan dan etika**: lihat di bawah.
7. **Kesimpulan**: rekomendasi detektor per skenario (dekat/jauh, real-time, CPU saja).

### Hal yang perlu dijaga
- **Bedakan skor keyakinan dan akurasi.** Laporkan recall, precision, F1, AP (deteksi) serta akurasi rank-1,
  DIR, FAR, FRR (pengenalan). Jangan menulis "akurasi YOLO 87%" dari angka demo.
- **Selalu sertakan interval kepercayaan** (Wilson 95% / bootstrap). Subjek sedikit, jadi klaim harus hati-hati
  ("cenderung lebih tinggi", bukan "terbukti lebih baik") bila interval tumpang tindih.
- **Sebut perangkat keras secara jujur.** MediaPipe berjalan di GPU (Metal), sedangkan Haar dan YOLO di CPU.
  Kecepatan antar-detektor tidak setara dan harus disebut (PRD §8.5).
- **Keterbatasan:** jumlah subjek kecil dan satu demografi; galeri dan uji LBPH dari sesi yang sama sehingga
  akurasi bisa terlalu optimistis; blur di E7 adalah simulasi; sudut pose nominal; tanpa *fine-tuning*.
- **Etika:** persetujuan tertulis (penelitian, publikasi, pengenalan terpisah), hanya kode pseudonim, data tidak
  dipublikasikan, model LBPH dihapus bila izin dicabut. Hanya tampilkan wajah peserta dengan `consent_publication = ya`.

### Referensi inti (sudah ada di PRD §17)
- Viola & Jones (2001) — Haar Cascade
- Bazarevsky dkk. (2019) — BlazeFace; Lugaresi dkk. (2019) — MediaPipe
- Jocher, Chaurasia, & Qiu (2023) — Ultralytics YOLOv8; repo derronqi/yolov8-face dan akanametov/yolo-face
- Yang dkk. (2016) — WIDER FACE (dataset latih YOLOv8-face)
- Ahonen, Hadid, & Pietikäinen (2006) — LBP untuk pengenalan wajah
- Phillips, Grother, & Micheals (2011) — evaluasi pengenalan wajah (DIR, FAR)
- Everingham dkk. (2010) — PASCAL VOC (definisi AP)
- Zuiderveld (1994) — CLAHE
- Wilson (1927) — interval kepercayaan proporsi

### Prompt pencarian paper (Elicit/Consensus)
1. Comparison of Haar cascade, MediaPipe, and YOLO for face detection accuracy and speed
2. Effect of camera distance and face size on face detection performance
3. Face detection performance under low illumination and CLAHE preprocessing
4. Robustness of face detectors to head pose, facial expression, and occlusion (masks, sunglasses)
5. YOLOv8 face detection evaluation on WIDER FACE or real-world images
6. LBPH face recognition accuracy under varying lighting and pose
7. Real-time face detection on CPU: lightweight models comparison
8. Effect of image resolution and motion blur on face detection

## 5. Mulai lagi di lokal

```bash
cd face-detection
git fetch origin
git checkout claude/face-recognition-yolo-crop-mmyiyt
git pull
source .venv/bin/activate
python -m pcdface selftest
pytest
claude                       # Claude Code lokal; minta dia membaca docs/CATATAN_SESI.md
```
