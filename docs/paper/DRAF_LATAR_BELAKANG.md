# Draf Paper — Kajian Pustaka dan Latar Belakang

**Topik usulan:** Perbandingan deteksi wajah OpenCV (Haar Cascade) dan YOLOv8-face, crop wajah manual vs otomatis, serta pengaruhnya pada pengenalan identitas LBPH.

**Status:** draf 12 Oktober 2026, diperbarui dengan rujukan 2024–2026 dan status indeks SINTA/Scopus (§1b). Ringkasan di bawah adalah parafrase kami dari abstrak. **Sebelum dikutip di naskah final, buka setiap paper, cocokkan angka dengan isi paper, dan lengkapi DOI, volume, serta halaman.** Detail penulis di daftar pustaka masih "dkk." karena hasil pencarian hanya memuat penulis pertama.

---

## Langkah demi langkah (alur yang dipakai)

| # | Langkah | Yang dikerjakan di draf ini |
|---|---|---|
| 1 | Cari di Elicit: *face detection using OpenCV vs YOLO* | Elicit tidak tersedia di sesi ini. Pencarian setara memakai **Consensus** (mesin pencari paper ilmiah, 400 juta+ paper) dengan 4 kueri: (a) *comparison of Haar cascade and YOLO for face detection accuracy and speed*; (b) *Viola-Jones Haar cascade limitations pose illumination compared with deep learning*; (c) *YOLOv8 face detection performance evaluation*; (d) *effect of face bounding box and cropping errors on face recognition accuracy LBPH*. Kalau ingin tetap memakai Elicit, ketik kueri yang sama lalu bandingkan hasilnya dengan tabel di §1. |
| 2 | Ambil abstrak sebagai rujukan pendahuluan | Dari ±40 hasil, dipilih 13 paper yang langsung relevan (§1). Paper yang angkanya janggal di abstrak tidak dipakai (lihat catatan §1). |
| 3 | Pilih tahun terbaru | Filter tahun ≥ 2021 (kueri c: ≥ 2023). Rentang terpilih: 2021–2026. Rujukan klasik (Viola & Jones, 2001; Ahonen dkk., 2006) tetap dipakai untuk landasan teori. |
| 4 | Ringkas tiap abstrak menjadi **masalah – metode – hasil** | §1, satu baris per paper. |
| 5 | Gabungkan masalah vs masalah, metode vs metode, hasil vs hasil | §2. |
| 6 | Beri sitasi | Gaya APA (nama, tahun) di teks; daftar pustaka di §6. |
| 7 | Narasi latar belakang untuk penelitian usulan | §4. Perbandingan crop manual vs otomatis dan perbandingan akurasi juga masuk ke §3 dan §5. |

---

## 1. Matriks masalah – metode – hasil (per paper)

| Kode | Paper | Masalah | Metode | Hasil |
|---|---|---|---|---|
| P1 | Andrean dkk. (2024) | Klasifikasi kantuk pengemudi dari seluruh wajah bisa bias; perlu daerah minat (ROI) mata yang bersih | Haar Cascade vs YOLO-face untuk memotong ROI kedua mata, lalu *decision tree* (grid search, 10-fold CV) pada ±41 ribu citra | ROI dari YOLO-face lebih berkualitas; akurasi 98,54% (YOLO-face) vs 98,03% (Haar). YOLO-face jauh lebih lama memproses data |
| P2 | Dewanto dkk. (2024) | Pengenalan wajah bermasker secara *real-time* | YOLO vs Haar Cascade + CNN; metrik akurasi, *framerate*, F1 | Haar + CNN akurasi 99,3% vs YOLO 79,3%; YOLO unggul di precision dan recall; Haar + CNN sedikit lebih lambat |
| P3 | Pebrianto dkk. (2022) | YOLOv3 berat untuk perangkat berdaya rendah | YOLOv3 vs YOLOv3-tiny dilatih data wajah; diuji per jarak wajah dan jumlah wajah, citra RGB dan abu-abu | YOLOv3 ±95% pada wajah dekat dan banyak wajah; YOLOv3-tiny 87% pada wajah dekat, tetapi jauh lebih cepat (104 vs 30 FPS di GPU; 8,7 vs 1,6 FPS di CPU) |
| P4 | Adouani dkk. (2022) | Perlu detektor wajah yang akurat sekaligus cepat pada video spontan | Haar, LBP *cascade*, HOG + SVM, dan CNN pada video RECOLA dan DEAP | CNN paling akurat (99,99% DEAP; 84,23% RECOLA) tetapi paling lambat (±2 FPS); LBP paling cepat (±26–34 FPS) |
| P5 | Abdulhussien dkk. (2022) | Viola-Jones masih banyak dipakai, tetapi keterbatasannya perlu dipetakan | Implementasi dan serangkaian uji Viola-Jones | Kelebihan: sederhana, *real-time*, butuh daya komputasi rendah. Paper memetakan faktor yang menurunkan akurasi dan usulan perbaikannya |
| P6 | Saputra dkk. (2025) | Penerapan Viola-Jones dalam aplikasi GUI | GUI Matlab; 15 foto internet berisi satu dan banyak wajah | Rata-rata akurasi 89,86%; gagal pada wajah tertutup, tidak menghadap depan, dan cahaya redup |
| P7 | Feng dkk. (2021) | Detektor tradisional kesulitan pada wajah *in the wild* (skala, pose, oklusi) | Survei dan evaluasi detektor berbasis *deep learning* dari sisi FLOPs dan latensi | *Deep learning* memberi lompatan akurasi dengan biaya komputasi yang jauh lebih besar; panduan memilih detektor per aplikasi |
| P8 | Minaee dkk. (2021) | Fitur buatan tangan (Haar, HOG) kurang kuat untuk lingkungan tak terkontrol | Survei metode deteksi wajah *deep learning* dan dataset | Peralihan ke *deep learning* meningkatkan akurasi secara signifikan; tantangan tersisa: wajah kecil, oklusi, efisiensi |
| P9 | Yu dkk. (2022) | Wajah kecil dan wajah terhalang masih sulit bagi detektor satu tahap | YOLO-FaceV2 (berbasis YOLOv5) dengan modul *receptive field*, *attention*, dan *loss* untuk oklusi; diuji di WIDER FACE | Mengungguli YOLO dan variannya di subset *easy*, *medium*, dan *hard* WIDER FACE |
| P10 | Thakre dkk. (2026) | Deteksi wajah *real-time* di perangkat tepi berdaya rendah | YOLOv8 vs RetinaFace di Raspberry Pi 4; WIDER FACE dan FDDB; akurasi, latensi, FPS, daya | YOLOv8 lebih cepat dengan akurasi yang masih dapat diterima; RetinaFace lebih presisi tetapi latensinya lebih tinggi |
| P11 | Paul dkk. (2021) | Pengenalan wajah pada resolusi rendah | Viola-Jones (deteksi) + LBPH dengan CLAHE dan *alignment*; dataset sendiri | Akurasi LBPH 98,05% pada wajah 45 px tetapi 78,40% pada 15 px; toleh ±30° → 72–82% |
| P12 | Kapse dkk. (2022) | Cara mengambil ROI wajah memengaruhi pengenalan | Tiga cara membuat kotak ROI (min-max, *alignment* OpenCV, berbasis posisi mata) + CNN ringan pada dataset MUCT | ROI berbasis mata memberi akurasi pengenalan lebih tinggi daripada dua cara lain — **cara crop berpengaruh pada akurasi** |
| P13 | Zou dkk. (2021) | Kotak wajah dari anotator dan detektor berbeda-beda (terlalu besar atau kecil) | Detektor *landmark* yang tahan terhadap berbagai cara crop | Kotak yang tidak tepat menurunkan kinerja tahap berikutnya secara tajam; metode yang diusulkan mengurangi galat >6,5% |
| P14 | Luo dkk. (2021) | Kotak anotasi manual sering tidak sejajar dengan kotak prediksi | Kalibrasi kotak anotasi dengan kotak prediksi model (BDC) pada beberapa detektor dan dua benchmark | Ketidaksejajaran anotasi adalah sumber utama galat lokalisasi; kalibrasi menaikkan precision dan recall |

**Tidak dipakai:** Ramli (2024), *YOLOv7-Tiny and YOLOv8n Evaluation for Face Detection*. Angka di abstraknya tidak konsisten (mAP50-95 lebih besar daripada mAP50), jadi baru boleh dipakai setelah dicek di naskah lengkapnya.

## 1b. Rujukan terbaru 2024–2026 (Scopus dan SINTA)

Dicari ulang dengan tahun ≥ 2024. Dua kueri internasional memakai filter SJR, sehingga hasilnya hanya jurnal yang punya peringkat SCImago (terindeks Scopus). Dua kueri lain sengaja tanpa filter, untuk menjaring jurnal Indonesia (SINTA).

### Status indeks

| Kode | Jurnal / prosiding | Status | Dasar |
|---|---|---|---|
| P15 | Journal of Communications Software and Systems (JCOMSS) | **Scopus** (kuartil terakhir yang terlihat Q4) | Lolos filter SJR; situs jurnal menyatakan terindeks Scopus |
| P16 | IEEE Access | **Scopus** | Lolos filter SJR |
| P17 | Alexandria Engineering Journal | **Scopus** | Lolos filter SJR |
| P18 | IEEE Access | **Scopus** | Lolos filter SJR |
| P19 | Journal of Imaging (MDPI) | **Scopus** | Lolos filter SJR |
| P20 | Sci (MDPI) | **Scopus** | Lolos filter SJR |
| P21 | ICVEE 2024 (prosiding IEEE, UNESA) | IEEE Xplore/Scopus **belum terkonfirmasi** untuk edisi 2024 (edisi 2023 menyatakan Scopus) | Situs konferensi |
| P22 | International Journal on Computational Engineering | **Belum terkonfirmasi** | Tidak ditemukan di pencarian web |
| P23 | IJoICT (Telkom University) | SINTA **belum terkonfirmasi** (pernyataan 2022: sedang proses akreditasi) | Situs jurnal |
| P24 | Inform: Jurnal Ilmiah Bidang TIK (Unitomo) | **SINTA 2** (SK 295/C/C3/KPT/2026, Vol. 10 2025 – Vol. 14 2029) | Situs jurnal |
| P25 | ELECTRON Jurnal Ilmiah Teknik Elektro | **Belum dicek** | — |
| P26 | International Journal of Business Economics and Informatics | **Belum dicek** | — |
| P2 | Scientific Journal of Informatics (UNNES) | **SINTA 2**; Scopus tidak terkonfirmasi | Situs jurnal |
| P1 | Jurnal Media Informatika Budidarma | SINTA 3 menurut dokumen 2020; **peringkat terbaru belum terkonfirmasi** | Formulir review 2020 |

Sebelum dikutip, cek peringkat terbaru di [sinta.kemdiktisaintek.go.id](https://sinta.kemdiktisaintek.go.id) (cari nama jurnal atau ISSN) dan kuartil Scopus di [scimagojr.com](https://www.scimagojr.com). Peringkat berubah setiap tahun, dan dosen biasanya meminta bukti ini.

### Matriks masalah – metode – hasil (rujukan terbaru)

| Kode | Paper | Masalah | Metode | Hasil |
|---|---|---|---|---|
| P15 | Naser dkk. (2025), JCOMSS — Scopus | Ketahanan detektor terhadap toleh kepala (yaw) belum diuji sistematis karena dataset bersudut terukur langka | Haar Cascade vs MTCNN pada dataset UPM: 100 mahasiswa, toleh −90° s.d. +90°, masker, kacamata | MTCNN paling tahan di semua sudut (100%; masker 99,9%; kacamata 96,4%; keduanya 80%). Haar: 92,5%; 67,3%; 80,4%; 76,3% |
| P16 | Narayanan dkk. (2025), IEEE Access — Scopus | Deteksi dan penghitungan wajah pada kondisi citra sulit (noise, blur, cahaya, wajah hewan) | YOLOv9 dilatih WIDER FACE, diuji pada berbagai kualitas citra dan Raspberry Pi | Precision 86%, recall 62,8%, mAP 70,8% di WIDER FACE (15,2 ms). Kinerja turun pada noise dan blur, dan muncul deteksi palsu pada wajah hewan serta patung/poster |
| P17 | Sun dkk. (2025), Alexandria Eng. J. — Scopus | Wajah kecil di adegan kompleks sulit dideteksi | YOLOv8 dengan fusi multi-skala, *loss* yang dioptimalkan, dan pelatihan dengan derau Gaussian | 0,780 pada validasi WIDER FACE, di atas metode arus utama |
| P18 | Yalçin dkk. (2026), IEEE Access — Scopus | Variasi skala, oklusi, pose, dan kepadatan menyulitkan deteksi wajah *real-time* | YOLO11n dengan rancang ulang *backbone* dan *loss* DC-CIoU; WIDER FACE + FDDB | Varian CSPResNeXt paling akurat; DC-CIoU menaikkan recall pada subset sulit; YOLO11n tetap paling efisien |
| P19 | Chen dkk. (2025), J. Imaging — Scopus | Deteksi kelelahan pengemudi butuh deteksi wajah cepat dan akurat | YOLOv8 yang diperbaiki → **crop wajah** → 68 titik *landmark* (PFLD) → indikator mata/mulut/kepala | Akurasi dan efisiensi lebih baik di WIDER FACE dan YAWDD. Crop dari detektor menjadi masukan tahap berikutnya |
| P20 | Shobaki dkk. (2025), Sci — Scopus | Sistem menulis dengan tatapan mata lambat dan kurang andal | Membandingkan Haar, YOLOv8, SSD, Faster R-CNN, dan lainnya pada citra mata | Haar akurasi tertinggi (0,85); YOLOv8 hampir setara (0,83), jauh lebih cepat, dan modelnya terkecil |
| P21 | Ananda dkk. (2024), ICVEE (IEEE) | Deteksi banyak wajah di kelas (cahaya, oklusi, arah hadap beragam) | Haar Cascade, MTCNN, YOLOFace, RetinaFace pada foto kelas nyata; precision, recall, F1, waktu | YOLOFace nyaris sempurna. **Haar: precision 0,89 tetapi recall 0,25** (banyak wajah terlewat), walau paling cepat |
| P22 | Irfiansyah dkk. (2025), Int. J. Computational Eng. | Akurasi deteksi turun pada wajah dari berbagai sudut | YOLO vs Haar Cascade pada berbagai sudut, ekspresi, dan cahaya; citra dan video | YOLO konsisten 100% di semua kondisi. Haar turun tajam pada sudut ekstrem −90° (mis. 36,96% pada citra), tetapi lebih cepat |
| P23 | Sartika dkk. (2026), IJoICT | Pengenalan banyak wajah dalam satu frame secara *real-time* | Haar Cascade + LBPH (OpenCV, Python); variasi cahaya, sudut, jumlah wajah | ±90% pada cahaya normal; >80% pada cahaya redup atau wajah menyamping; 40–60 ms per wajah |
| P24 | Kudadiri dkk. (2026), Inform — SINTA 2 | Kunci pintu konvensional rentan disalahgunakan | Haar + LBPH dengan ESP32-CAM; 3 peserta, 7 uji | Berhasil membuka 6/7 (85,71%), respons ±2,2 detik; kasus tak dikenal ditolak dengan benar |
| P25 | Elbert dkk. (2025), ELECTRON | Metode deteksi wajah memengaruhi kinerja pengenalan | Haar, dlib, MediaPipe sebagai pemotong wajah → CNN pengenal | CNN dengan crop Haar justru tertinggi (akurasi 98%, F1 97,99%) — **sumber crop memengaruhi pengenalan** |
| P26 | Irawan dkk. (2025), IJBEI | Batas kerja pipeline klasik jarang dilaporkan | Haar + LBPH (`cv2.face`) pada 50 subjek, ambang *confidence* ≤ 50 | Akurasi 89,7%; turun tajam bila toleh > 30° atau oklusi > 40% |

### Dampaknya pada sintesis (§2)

- **Masalah.** Rujukan terbaru mempertegas tiga masalah lama: pose ekstrem (P15, P22), wajah kecil dan kualitas citra (P16, P17, P18), serta banyak wajah dalam satu frame (P21, P23).
- **Metode.** Tiga paper sudah memakai precision, recall, dan F1 (P21, P22, P16). Belum ada yang mengukur kecocokan crop terhadap crop manual. Dua paper membandingkan sumber crop untuk tahap berikutnya (P25, P19), tetapi tanpa crop manual sebagai acuan.
- **Hasil.** Polanya konsisten: YOLO lebih tinggi recall-nya dan lebih tahan pose (P21, P22), sedangkan Haar lebih cepat. Pada objek atau tugas tertentu Haar bisa menyamai bahkan mengungguli (P20, P25). Pipeline Haar + LBPH masih dipakai luas di Indonesia dengan akurasi ±85–92% (P23, P24, P26), tetapi turun tajam pada toleh > 30° (P26).

---

## 2. Sintesis: masalah vs masalah, metode vs metode, hasil vs hasil

### 2.1 Masalah

- **Kondisi tak terkontrol.** Semua paper sepakat bahwa variasi pose, cahaya, oklusi, dan skala wajah adalah sumber kegagalan utama, terutama bagi detektor klasik (P5, P6, P7, P8). Untuk YOLO, masalah yang tersisa bergeser ke wajah sangat kecil dan wajah terhalang (P9).
- **Akurasi vs komputasi.** Masalah kedua adalah tukar-tambah antara akurasi dan beban komputasi. Perangkat berdaya rendah dan aplikasi *real-time* menuntut detektor ringan (P3, P4, P10), sedangkan detektor *deep learning* menuntut daya komputasi besar (P7).
- **Kualitas crop untuk tahap berikutnya.** Kelompok paper ketiga menyoroti bahwa hasil deteksi dipakai lagi: sebagai ROI klasifikasi (P1), sebagai masukan pengenal identitas (P2, P11, P12), atau sebagai masukan detektor *landmark* (P13). Kotak yang tidak tepat merusak tahap itu, dan kotak manual pun tidak selalu sejajar dengan kotak detektor (P14).

### 2.2 Metode

- **Pembanding.** Perbandingan klasik vs *deep learning* dilakukan dengan Haar vs YOLO (P1, P2), Haar/LBP/HOG vs CNN (P4), atau antarvarian YOLO (P3, P10). Hanya P1 dan P2 yang secara langsung membandingkan Haar dengan YOLO, dan keduanya menilai **akurasi tugas akhir** (kantuk, masker), bukan ketepatan kotak deteksi itu sendiri.
- **Data.** Sebagian memakai benchmark besar (WIDER FACE, FDDB: P9, P10), sebagian memakai dataset sendiri berukuran kecil sampai sedang (P1, P2, P3, P6, P11).
- **Metrik.** Metrik yang dilaporkan beragam: "akurasi" (P1, P2, P6), precision/recall/F1 (P2), mAP (P3, P9), FPS dan latensi (P3, P4, P10). Hampir tidak ada yang melaporkan **interval kepercayaan**, dan tidak ada yang mengukur **kecocokan kotak (IoU) detektor terhadap crop manual** sebagai hasil penelitian tersendiri.
- **Crop.** Pengaruh cara crop diukur pada pengenal identitas (P12) dan detektor *landmark* (P13), tetapi tidak pada perbandingan Haar vs YOLO.

### 2.3 Hasil

- **Akurasi.** Detektor *deep learning* (CNN, YOLO, RetinaFace) umumnya lebih akurat daripada Haar pada kondisi tak terkontrol (P4, P7, P8). P1 menunjukkan ROI dari YOLO-face lebih baik daripada ROI Haar.
- **Ada pengecualian.** Pada tugas tertentu, Haar + CNN justru lebih akurat daripada YOLO (P2). Jadi keunggulan YOLO tidak otomatis; ia bergantung pada tugas, data, dan cara mengukurnya.
- **Kecepatan.** Detektor klasik dan varian ringan jauh lebih cepat (P1, P3, P4, P10).
- **Kualitas crop.** Kualitas crop terbukti memengaruhi akurasi pengenalan (P11: ukuran wajah; P12: cara membuat ROI). Perbedaan konvensi kotak antara manusia dan detektor juga nyata (P13, P14).

---

## 3. Perbandingan akurasi menurut literatur

Angka di bawah **tidak dapat dibandingkan langsung**, karena dataset, tugas, dan definisi "akurasi"-nya berbeda-beda. Tabel ini menunjukkan rentang hasil yang dilaporkan, dan sekaligus memperlihatkan perlunya evaluasi pada data dan metrik yang sama.

| Paper | Metode | Tugas / data | Akurasi / metrik | Kecepatan |
|---|---|---|---|---|
| P1 | Haar Cascade | ROI mata → klasifikasi kantuk | Akurasi 98,03% | Lebih cepat |
| P1 | YOLO-face | ROI mata → klasifikasi kantuk | Akurasi 98,54% | Jauh lebih lambat |
| P2 | Haar + CNN | Wajah bermasker | Akurasi 99,3% | Sedikit lebih lambat |
| P2 | YOLO | Wajah bermasker | Akurasi 79,3%; precision dan recall lebih tinggi | Lebih cepat |
| P3 | YOLOv3 | Deteksi wajah, wajah dekat / banyak wajah | ±95% | 1,6 FPS (CPU) |
| P3 | YOLOv3-tiny | Deteksi wajah, wajah dekat | 87% | 8,7 FPS (CPU) |
| P4 | CNN | Video DEAP / RECOLA | Tingkat deteksi 99,99% / 84,23% | ±2 FPS |
| P4 | LBP *cascade* | Video DEAP / RECOLA | Lebih rendah dari CNN | ±26–34 FPS |
| P6 | Viola-Jones (Haar) | 15 foto internet | Akurasi rata-rata 89,86% | — |
| P11 | Viola-Jones + LBPH | Pengenalan, wajah 45 px / 15 px | 98,05% / 78,40% | *Real-time* |
| P15 | Haar Cascade | Toleh −90°…+90° / masker / kacamata | 92,5% / 67,3% / 80,4% | — |
| P15 | MTCNN | Toleh −90°…+90° / masker / kacamata | 100% / 99,9% / 96,4% | — |
| P16 | YOLOv9 | WIDER FACE | P 0,86; R 0,628; mAP 0,708 | 15,2 ms |
| P20 | Haar | Deteksi mata | Akurasi 0,85 | Lebih lambat |
| P20 | YOLOv8 | Deteksi mata | Akurasi 0,83 | Jauh lebih cepat |
| P21 | Haar Cascade | Banyak wajah di kelas | **P 0,89; R 0,25** | Tercepat (0,54 s) |
| P21 | YOLOFace | Banyak wajah di kelas | P, R, F1 ≈ 1,00 di sebagian besar skenario | Lebih lambat |
| P22 | YOLO | Multi-pose, ekspresi, cahaya | 100% di semua kondisi | Lebih lambat |
| P22 | Haar Cascade | Multi-pose; sudut −90° | Turun tajam (mis. 36,96%) | Lebih cepat |
| P23 | Haar + LBPH | Banyak wajah, *real-time* | ±90% (normal); >80% (redup / menyamping) | 40–60 ms/wajah |
| P26 | Haar + LBPH | 50 subjek, *indoor* | 89,7% | CPU |

**Celah yang terlihat dari tabel:** setiap paper memakai metrik sendiri, dan sebagian besar tidak menyebut interval kepercayaan maupun ambang IoU. Penelitian kami menjawabnya dengan satu protokol yang sama untuk kedua detektor (§5).

---

## 4. Narasi latar belakang (siap disunting)

> **Catatan:** paragraf di bawah sudah memuat sitasi. Angka hasil penelitian kami sendiri **sengaja belum diisi**. Isi setelah `eval-folder` dijalankan pada data asli, dan ambil angkanya dari `results/e9/RINGKASAN_E9.md`.

Deteksi wajah adalah tahap pertama hampir semua sistem analisis wajah, mulai dari pengenalan identitas sampai pemantauan pengemudi (Feng dkk., 2021; Minaee dkk., 2021). Metode klasik Viola–Jones, yang tersedia sebagai Haar Cascade di pustaka OpenCV, masih banyak dipakai karena sederhana, berjalan *real-time*, dan hanya membutuhkan daya komputasi rendah (Abdulhussien dkk., 2022). Namun metode ini dirancang untuk wajah tampak depan. Akurasinya turun pada wajah yang tertutup, tidak menghadap kamera, atau berada di cahaya redup (Saputra dkk., 2025). Sejak berkembangnya *deep learning*, detektor berbasis jaringan saraf konvolusional memberi lompatan akurasi pada kondisi tak terkontrol, dengan biaya komputasi yang jauh lebih besar (Feng dkk., 2021; Minaee dkk., 2021). Keluarga YOLO menjadi pilihan populer karena menyeimbangkan kecepatan dan akurasi sebagai detektor satu tahap (Yu dkk., 2022), dan versi terbarunya terus dikembangkan untuk deteksi wajah (Sun dkk., 2025; Yalçin dkk., 2026). Meski demikian, wajah kecil, wajah terhalang, dan citra berkualitas rendah masih menjadi tantangan (Yu dkk., 2022; Narayanan dkk., 2025), dan varian yang lebih ringan mengorbankan sebagian akurasi demi kecepatan (Pebrianto dkk., 2022; Thakre dkk., 2026).

Penelitian terbaru yang membandingkan keduanya secara langsung cenderung menunjukkan keunggulan YOLO dalam menemukan wajah. Pada foto kelas berisi banyak wajah, YOLOFace mencapai precision dan recall yang hampir sempurna, sedangkan Haar Cascade hanya menemukan seperempat wajah (recall 0,25) meskipun paling cepat (Ananda dkk., 2024). Pada wajah dari berbagai sudut, YOLO konsisten mendeteksi wajah, sedangkan akurasi Haar Cascade turun tajam pada sudut ekstrem (Irfiansyah dkk., 2025). Kelemahan yang sama terlihat ketika Haar Cascade dibandingkan dengan detektor CNN lain pada toleh hingga 90°, masker, dan kacamata (Naser dkk., 2025). Meski demikian, hasilnya belum seragam. Andrean dkk. (2024) mendapati bahwa daerah mata yang dipotong YOLO-face lebih berkualitas daripada hasil Haar Cascade, tetapi waktu prosesnya jauh lebih lama. Sebaliknya, Dewanto dkk. (2024) melaporkan bahwa pada wajah bermasker, kombinasi Haar Cascade dan CNN justru mencapai akurasi lebih tinggi daripada YOLO, meskipun YOLO unggul dalam precision dan recall. Kedua penelitian itu menilai akurasi **tugas akhir** (klasifikasi kantuk atau masker), bukan ketepatan detektor dalam menemukan dan membatasi wajah. Perbandingan yang lebih luas antara detektor klasik dan CNN pada video juga menunjukkan tukar-tambah yang sama: CNN paling akurat tetapi paling lambat (Adouani dkk., 2022). Metrik yang dipakai pun berbeda-beda (akurasi, mAP, F1, FPS) dan jarang disertai interval kepercayaan, sehingga hasil antarpenelitian sulit dibandingkan.

Persoalan lain yang jarang diukur adalah **kualitas potongan (crop) wajah** yang dihasilkan detektor. Crop inilah yang diteruskan ke tahap berikutnya, misalnya pengenalan identitas. Kapse dkk. (2022) menunjukkan bahwa cara menentukan kotak wajah memengaruhi akurasi pengenalan, dan Elbert dkk. (2025) mendapati bahwa pengenal CNN memberi akurasi berbeda bergantung pada detektor yang memotong wajahnya (Haar Cascade, dlib, atau MediaPipe). Pipeline Haar Cascade + LBPH sendiri masih banyak diterapkan dengan akurasi sekitar 85–92% (Sartika dkk., 2026; Irawan dkk., 2025; Kudadiri dkk., 2026), tetapi akurasinya turun tajam ketika wajah menoleh lebih dari 30° atau tertutup lebih dari 40% (Irawan dkk., 2025). Paul dkk. (2021) menunjukkan bahwa akurasi LBPH turun tajam ketika wajah yang dipotong berukuran kecil. Kotak buatan anotator dan kotak buatan detektor juga mengikuti konvensi yang berbeda: kotak yang terlalu besar atau terlalu kecil menurunkan kinerja tahap berikutnya (Zou dkk., 2021), dan kotak anotasi manual sering tidak sejajar dengan kotak prediksi model (Luo dkk., 2021). Karena itu, perbandingan detektor sebaiknya tidak berhenti pada pertanyaan "apakah wajah ditemukan". Perbandingan juga perlu mengukur seberapa mirip crop otomatis dengan crop manual, dan apakah perbedaan itu mengubah hasil pengenalan.

Berdasarkan celah tersebut, penelitian ini membandingkan dua pendekatan deteksi wajah yang sama-sama dijalankan melalui pustaka OpenCV: Haar Cascade (Viola & Jones, 2001) sebagai metode klasik dan YOLOv8-face melalui modul DNN OpenCV sebagai metode *deep learning*. Keduanya dievaluasi terhadap **crop manual** sebagai *ground truth* pada foto wajah peserta yang telah memberikan persetujuan tertulis. Penelitian ini mengukur (1) akurasi deteksi dengan precision, recall, F1, dan *average precision* pada ambang IoU 0,5, disertai interval kepercayaan 95%; (2) kecocokan crop otomatis terhadap crop manual melalui IoU, pergeseran pusat, dan rasio ukuran kotak; serta (3) pengaruh sumber crop (manual, Haar, YOLO) terhadap akurasi pengenalan identitas dengan LBPH (Ahonen dkk., 2006). Hasilnya diharapkan menjadi rekomendasi pemilihan detektor untuk sistem pengenalan wajah sederhana berbasis OpenCV.

### Rumusan masalah (turunan langsung dari narasi)

1. Bagaimana perbandingan akurasi deteksi (precision, recall, F1, AP) Haar Cascade dan YOLOv8-face terhadap crop manual?
2. Seberapa mirip crop wajah otomatis dari kedua detektor dengan crop manual (IoU, pergeseran pusat, rasio ukuran)?
3. Apakah akurasi pengenalan LBPH berbeda bila wajah di-crop secara manual, oleh Haar Cascade, atau oleh YOLOv8-face?

---

## 5. Perbandingan yang diukur sendiri (rencana tabel Hasil)

Semua angka diambil dari `python -m pcdface eval-folder` (E9). Urutan pengerjaannya sesuai arahan:

1. `annotate --folder data/DATASET-MUSANG-PCD`: buat **crop manual** untuk semua foto (aturan kotak PRD §7).
2. `eval-folder --input data/DATASET-MUSANG-PCD --detectors haar`: jalankan **model OpenCV** dulu.
3. `eval-folder --input data/DATASET-MUSANG-PCD --detectors yolo_n`: jalankan **model YOLO**. Setelah itu tabel otomatis membandingkan keduanya.

| Tabel di paper | Berkas | Isi |
|---|---|---|
| Tabel 1 — Akurasi deteksi | `results/e9/e9_deteksi.md` + `e9_selisih_detektor.md` | TP, FP, FN, precision, recall, F1, AP, rerata IoU, waktu; selisih Haar − YOLO dengan bootstrap berpasangan |
| Tabel 2 — Crop manual vs otomatis | `results/e9/e9_crop.md` | % crop dengan IoU ≥ 0,5, median IoU, geser pusat, rasio lebar/tinggi, aspek kotak |
| Tabel 3 — Pengenalan LBPH | `results/e9/e9_pengenalan.md` + `e9_pengenalan_selisih.md` | Akurasi rank-1 dan DIR untuk crop manual vs Haar vs YOLO (pembagian 70/30 per peserta) |
| Gambar | `grafik16`–`grafik19` di `results/e9/` | Batang P/R/F1/AP, sebaran IoU, konvensi kotak, akurasi pengenalan |
| Gambar contoh crop | `results/e9/banding_crop/` | Crop manual, Haar, dan YOLO berdampingan. **Jangan masukkan ke paper**: semua peserta belum mengizinkan publikasi wajah (`consent_publication = tidak`). Pakai grafik 18 (konvensi kotak), yang tidak menampilkan wajah |

**Aturan penulisan hasil** (PRD §8.6): sebut satu detektor "lebih baik" hanya bila interval selisihnya tidak memuat 0. Dengan 4 peserta, intervalnya akan lebar, jadi gunakan frasa seperti "cenderung lebih tinggi" bila interval tumpang tindih.

---

## 6. Daftar pustaka (APA, perlu dilengkapi)

Rujukan dari pencarian (2021–2026):

- Abdulhussien, D. M., dkk. (2022). Evaluation study of face detection by Viola-Jones algorithm. *International Journal of Health Sciences*. [tautan](https://consensus.app/papers/details/b5ca9fd30c21509bb2034c0a356d164c/?utm_source=claude_desktop)
- Adouani, A., dkk. (2022). A comparison of face detection methods using spontaneous videos. *Multimedia Tools and Applications*. [tautan](https://consensus.app/papers/details/f6859ddd034a5eb09dbab7b99db1e8fc/?utm_source=claude_desktop)
- Andrean, M. N., dkk. (2024). Comparing Haar Cascade and YOLOFACE for region of interest classification in drowsiness detection. *Jurnal Media Informatika Budidarma*. [tautan](https://consensus.app/papers/details/7597cbe5798552e399cff5523eb76698/?utm_source=claude_desktop)
- Dewanto, M. R., dkk. (2024). YOLO vs. CNN algorithms: A comparative study in masked face recognition. *Scientific Journal of Informatics*. [tautan](https://consensus.app/papers/details/b6f64b2d735f530fa0134da38bddb66d/?utm_source=claude_desktop)
- Feng, Y., dkk. (2021). Detect faces efficiently: A survey and evaluations. *IEEE Transactions on Biometrics, Behavior, and Identity Science*. [tautan](https://consensus.app/papers/details/f9f63f814cc651fa94676a80464147bf/?utm_source=claude_desktop)
- Kapse, M. N., dkk. (2022). Eye-referenced dynamic bounding box for face recognition using light convolutional neural network. *Intelligent Decision Technologies*. [tautan](https://consensus.app/papers/details/14d936c37a995f5abfd3e902a4b3553a/?utm_source=claude_desktop)
- Luo, S., dkk. (2021). BDC: Bounding-box deep calibration for high performance face detection. *IET Computer Vision*. [tautan](https://consensus.app/papers/details/11b99664cdf15a028b9d6fd9b49b0821/?utm_source=claude_desktop)
- Minaee, S., dkk. (2021). Going deeper into face detection: A survey. *arXiv*. [tautan](https://consensus.app/papers/details/2835dc1a4f965e0aabf792d37234e15f/?utm_source=claude_desktop)
- Paul, K., dkk. (2021). An improved real-time face recognition system at low resolution based on local binary pattern histogram algorithm and CLAHE. *arXiv*. [tautan](https://consensus.app/papers/details/a438805f42415f47afae4f61acd79bc3/?utm_source=claude_desktop)
- Pebrianto, W., dkk. (2022). YOLO method analysis and comparison for real-time human face detection. *11th EECCIS*. [tautan](https://consensus.app/papers/details/af9b2e8ff88f5013bdcb91ddaf7dcfe1/?utm_source=claude_desktop)
- Saputra, S., dkk. (2025). Graphical user interface (GUI) for face detection using Viola-Jones algorithm. *Bincang Sains dan Teknologi*. [tautan](https://consensus.app/papers/details/80aba45254045e78be191c0a1aa0e816/?utm_source=claude_desktop)
- Thakre, D., dkk. (2026). Performance evaluation of YOLOv8 and RetinaFace for real-time face detection on edge devices. *IJAICET*. [tautan](https://consensus.app/papers/details/2d71b96ab4f75030adb816fde99e0d73/?utm_source=claude_desktop)
- Yu, Z.-P., dkk. (2022). YOLO-FaceV2: A scale and occlusion aware face detector. *arXiv*. [tautan](https://consensus.app/papers/details/1bd4ae38da3055509941e20448ea01b5/?utm_source=claude_desktop)
- Zou, X., dkk. (2021). Towards unconstrained facial landmark detection robust to diverse cropping manners. *IEEE Transactions on Circuits and Systems for Video Technology*. [tautan](https://consensus.app/papers/details/6ee8b8447a8b52a9ba73a4bd75841c13/?utm_source=claude_desktop)

Rujukan terbaru 2024–2026 (§1b; status indeks di tabel §1b):

- Ananda, G. F., dkk. (2024). Comparative analysis of multi-face detection methods in classroom environments: Haar Cascade, MTCNN, YOLOFace, and RetinaFace. *ICVEE 2024* (IEEE). [tautan](https://consensus.app/papers/details/2804ebd616cc5538b214ff5946d7d059/?utm_source=claude_desktop)
- Chen, C., dkk. (2025). Lightweight and real-time driver fatigue detection based on MG-YOLOv8 with facial multi-feature fusion. *Journal of Imaging*. [tautan](https://consensus.app/papers/details/c09c336473125d349b77c852ccf80b9c/?utm_source=claude_desktop)
- Elbert, dkk. (2025). Comparative analysis of Haar Cascade classifier, Dlib, and Mediapipe for face recognition. *ELECTRON Jurnal Ilmiah Teknik Elektro*. [tautan](https://consensus.app/papers/details/401fa2f4d98b50bfa13a62d233a41aff/?utm_source=claude_desktop)
- Irawan, W., dkk. (2025). Design and implementation of a real-time face recognition system using OpenCV and LBPH. *International Journal of Business Economics and Informatics*. [tautan](https://consensus.app/papers/details/1487812e82d65e3d9939baff4238fd0e/?utm_source=claude_desktop)
- Irfiansyah, M. F., dkk. (2025). Comparative analysis of multi-pose face detection between YOLO and Haar Cascade classifier. *International Journal on Computational Engineering*. [tautan](https://consensus.app/papers/details/bb857f44855054829fbd675d431bad82/?utm_source=claude_desktop)
- Kudadiri, Z. H., dkk. (2026). Hybrid face recognition system using Haar Cascade and local binary pattern histogram for automatic smart door access. *Inform: Jurnal Ilmiah Bidang Teknologi Informasi dan Komunikasi*. [tautan](https://consensus.app/papers/details/f844dfac0c8d5d719170d990cb6ccc99/?utm_source=claude_desktop)
- Narayanan, S. P., dkk. (2025). YOLOv9-based human face detection and counting under human-animal faces, complex imaging environments, and image qualities. *IEEE Access*. [tautan](https://consensus.app/papers/details/4e7962c15d6555c5846583c344b3effc/?utm_source=claude_desktop)
- Naser, O. A., dkk. (2025). Comparative analysis of MTCNN and Haar Cascades for face detection in images with variation in yaw poses and facial occlusions. *Journal of Communications Software and Systems*. [tautan](https://consensus.app/papers/details/abf46fd39cd65cd1ba0a2a0d2ad5fc82/?utm_source=claude_desktop)
- Sartika, D. A., dkk. (2026). An experiment using the Haar-Cascade and LBPH algorithms for real-time recognition of multiple faces in a single frame. *International Journal on Information and Communication Technology (IJoICT)*. [tautan](https://consensus.app/papers/details/95af71b3ea0f56fa9401fbc0e878cb36/?utm_source=claude_desktop)
- Shobaki, W. A., dkk. (2025). A comparative study of YOLO, SSD, Faster R-CNN, and more for optimized eye-gaze writing. *Sci*. [tautan](https://consensus.app/papers/details/81b457a74bcb5532a598e2563fce5f1d/?utm_source=claude_desktop)
- Sun, L., dkk. (2025). Intelligent monitoring of small target detection using YOLOv8. *Alexandria Engineering Journal*. [tautan](https://consensus.app/papers/details/a8468a5051505fefbca698010a69af35/?utm_source=claude_desktop)
- Yalçin, A. Ş., dkk. (2026). Detector redesign and distance-corrected CIoU loss for YOLO11-based face detection. *IEEE Access*. [tautan](https://consensus.app/papers/details/c34ac10b390f571e98fa18f1ae5550f5/?utm_source=claude_desktop)

Rujukan landasan teori (sudah ada di PRD §17):

- Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. *CVPR*.
- Ahonen, T., Hadid, A., & Pietikäinen, M. (2006). Face description with local binary patterns: Application to face recognition. *IEEE TPAMI*.
- Yang, S., Luo, P., Loy, C. C., & Tang, X. (2016). WIDER FACE: A face detection benchmark. *CVPR*.
- Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8* [perangkat lunak].
- Everingham, M., dkk. (2010). The PASCAL Visual Object Classes (VOC) challenge. *IJCV*.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *JASA*.
