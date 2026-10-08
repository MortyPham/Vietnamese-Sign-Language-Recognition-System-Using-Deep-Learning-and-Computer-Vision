# Vietnamese-Sign-Language-Recognition-System-Using-Deep-Learning-and-Computer-Vision
This is the code and data for the research project Vietnamese Sign Language Recognition System Using Deep Learning and Computer Vision. Our team consists of Phạm Ngọc Minh, Phan Nhật Quân, Vũ Anh Thư.

## VSL Studio web interface

The redesigned interface has separate Home, Upload and Live Webcam pages,
larger typography, explicit upload confirmation and native camera preview.
It recognizes 22 trained letters and 60 phrases using the included checkpoints.
Live capture finishes after six seconds of visible stillness. Single-word
recognition becomes available when a trained word checkpoint is supplied.

With Python 3.10 or 3.11 installed, run from the repository root:

```powershell
cd web
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe test.py
```

Open **http://127.0.0.1:8000**. Model checkpoints are loaded from the repository's
`weights/` folder automatically. For phrase demonstrations, choose
**Sentences → Explore supported signs**; verified public videos are currently
available for **Xin chào**, **Cảm ơn** and **Bạn tên là gì?**

See [web/README.md](web/README.md) for inputs, model limitations, video sources,
word-model setup and verification. The original training notebooks, datasets
and model weights remain available in their existing folders.

To run the behavioral tests from `web/`:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
