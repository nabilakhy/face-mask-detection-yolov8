import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


HIJAU = (0, 200, 0)
MERAH = (255, 0, 0)
KUNING = (0, 190, 255)
PUTIH = (255, 255, 255)
HITAM = (0, 0, 0)

PETA_STATUS = {
    "with_mask": ("Masker Terdeteksi", HIJAU),
    "mask": ("Masker Terdeteksi", HIJAU),
    "without_mask": ("Peringatan: Gunakan Masker", MERAH),
    "no-mask": ("Peringatan: Gunakan Masker", MERAH),
    "incorrectly_worn_mask": ("Masker Tidak Tepat", KUNING),
    "mask_weared_incorrect": ("Masker Tidak Tepat", KUNING),
}

PRIORITAS = {
    MERAH: 3,
    KUNING: 2,
    HIJAU: 1,
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def petakan_status(nama_kelas: str):
    kunci = nama_kelas.strip().lower().replace(" ", "_")

    if kunci in PETA_STATUS:
        return PETA_STATUS[kunci]

    if "without" in kunci or kunci.startswith("no"):
        return "Peringatan: Gunakan Masker", MERAH

    if "incorrect" in kunci:
        return "Masker Tidak Tepat", KUNING

    return nama_kelas, HIJAU


def gambar_label(frame, teks, x, y, warna, skala=0.6, tebal=2):
    (lebar_teks, tinggi_teks), baseline = cv2.getTextSize(
        teks,
        FONT,
        skala,
        tebal,
    )

    y = max(y, tinggi_teks + 8)

    cv2.rectangle(
        frame,
        (x, y - tinggi_teks - baseline - 4),
        (x + lebar_teks + 8, y),
        warna,
        thickness=-1,
    )

    cv2.putText(
        frame,
        teks,
        (x + 4, y - baseline - 2),
        FONT,
        skala,
        PUTIH,
        tebal,
        cv2.LINE_AA,
    )


def gambar_banner(frame, teks, warna):
    _, lebar_frame = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (lebar_frame, 58),
        warna,
        thickness=-1,
    )

    cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0,
        frame,
    )

    cv2.putText(
        frame,
        teks,
        (14, 39),
        FONT,
        0.9,
        PUTIH,
        2,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Deteksi masker real-time dengan YOLOv8"
    )

    parser.add_argument(
        "--model",
        default="best.pt",
        help="Path file model",
    )

    parser.add_argument(
        "--source",
        default="0",
        help="Index webcam atau path video",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Nilai minimum confidence",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Ukuran input model",
    )

    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(
            f"[ERROR] Model '{args.model}' tidak ditemukan.\n"
            "Pastikan file best.pt berada di folder yang sama dengan program."
        )

    print(f"[INFO] Memuat model dari {args.model}")

    model = YOLO(args.model)
    nama_kelas = model.names

    print(
        f"[INFO] Kelas yang dikenali model: "
        f"{list(nama_kelas.values())}"
    )

    sumber = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(sumber)

    if not cap.isOpened():
        raise SystemExit(
            f"[ERROR] Tidak dapat membuka sumber video '{args.source}'. "
            "Pastikan kamera tersedia dan tidak sedang digunakan aplikasi lain."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(
        "[INFO] Program berjalan. "
        "Tekan 'q' untuk keluar dan 's' untuk menyimpan screenshot."
    )

    waktu_sebelum = time.time()
    fps = 0.0
    n_screenshot = 0

    while True:
        berhasil, frame = cap.read()

        if not berhasil:
            print("[WARNING] Frame tidak dapat dibaca.")
            break

        if isinstance(sumber, int):
            frame = cv2.flip(frame, 1)

        hasil = model.predict(
            frame,
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )[0]

        status_terkumpul = []

        for box in hasil.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            id_kelas = int(box.cls[0])
            confidence = float(box.conf[0])

            teks_status, warna = petakan_status(
                nama_kelas[id_kelas]
            )

            status_terkumpul.append(
                (teks_status, warna)
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                warna,
                2,
            )

            gambar_label(
                frame,
                f"{teks_status} {confidence:.0%}",
                x1,
                y1 - 6,
                warna,
            )

        if status_terkumpul:
            teks_utama, warna_utama = max(
                status_terkumpul,
                key=lambda status: PRIORITAS[status[1]],
            )

            jumlah = Counter(
                teks for teks, _ in status_terkumpul
            )[teks_utama]

            if jumlah > 1:
                label_banner = (
                    f"{teks_utama} ({jumlah} orang)"
                )
            else:
                label_banner = teks_utama

            gambar_banner(
                frame,
                label_banner,
                warna_utama,
            )
        else:
            gambar_banner(
                frame,
                "Tidak ada wajah terdeteksi",
                (90, 90, 90),
            )

        waktu_sekarang = time.time()
        selisih_waktu = max(
            waktu_sekarang - waktu_sebelum,
            1e-6,
        )

        fps = 0.9 * fps + 0.1 * (1.0 / selisih_waktu)
        waktu_sebelum = waktu_sekarang

        posisi_fps = (
            14,
            frame.shape[0] - 16,
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            posisi_fps,
            FONT,
            0.6,
            HITAM,
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            posisi_fps,
            FONT,
            0.6,
            PUTIH,
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(
            "Deteksi Masker - YOLOv8",
            frame,
        )

        tombol = cv2.waitKey(1) & 0xFF

        if tombol == ord("q"):
            break

        if tombol == ord("s"):
            n_screenshot += 1
            nama_file = f"screenshot_{n_screenshot}.jpg"

            cv2.imwrite(
                nama_file,
                frame,
            )

            print(
                f"[INFO] Screenshot tersimpan: {nama_file}"
            )

    cap.release()
    cv2.destroyAllWindows()

    print("[INFO] Program selesai.")


if __name__ == "__main__":
    main()