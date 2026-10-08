# VSL Studio

A redesigned local workspace for Vietnamese Sign Language recognition. The
interface keeps the original navy, cyan, and violet palette, with self-hosted
Manrope typography, responsive layouts, explicit upload confirmation, native
camera preview, and optional recognition details.

## Run on Windows

For an existing local environment:

```powershell
.\.venv\Scripts\python.exe test.py
```

Open **http://127.0.0.1:8000**. The original `/home.html`, `/upload.html`, and
`/webcam.html` routes open separate Home, Upload, and Live Webcam pages. The
navigation stays visible, and recognition pages focus on their own input and
translation controls. For a fresh environment,
use Python 3.10 or 3.11, create `.venv`, and install `requirements.txt`.

In the GitHub repository, this app lives in `web/`. Run `cd web` first, then
follow the commands below. The app also finds the repository's existing
alphabet and sentence checkpoints in the parent `weights/` folder.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Recognition modes

| Mode | Input | Model and status |
| --- | --- | --- |
| Letters | JPG / PNG / WebP or held webcam sign | Project YOLO alphabet checkpoint; 22 trained letters |
| Single words | Image or held webcam sign | YOLO adapter implemented; trained word weights are not present |
| Sentences | MP4 / WebM / MOV / AVI or moving webcam sign | Existing ResNet1D checkpoint; 60 phrases plus idle |

### Uploads

1. Select the recognition level.
2. Choose or drop a file, or select a bundled alphabet sample.
3. Review the preview, then press **Translate sign**.

Selection never runs inference. Sentence mode requires a video rather than a
still photo. Files are limited to 30 MB, clips to 45 seconds, and images to
20 megapixels. Unreadable files receive actionable errors; no detected sign
produces a retry state rather than an invented translation.

### Webcam

Enable the camera, press **Start a sign**, then sign naturally and keep your
hands visible and still for six seconds. Movement restarts the countdown.
Tracking loss or a processing gap resets it. Results appear only after the
pause, and sampling stops until **Start another sign** is pressed.

The video preview runs natively in the browser, targeting 30 fps. Recognition
samples unmirrored JPEG frames at at most 6 fps and 640 pixels on the longest
edge, with one request in flight. Slow inference cannot queue camera frames.
Each session owns a MediaPipe tracker. The visible preview is mirrored;
handedness is corrected to the dataset's anatomical left/right convention.
Camera tracks stop on source/model changes, tab hiding, or page exit.

The 45-second capture limit is a safety stop with a retry message; it does not
split a continuous recording into fixed prediction windows. Blank/expired
sessions are removed on new-session creation.

### What the sentence model can do

The supplied checkpoint was trained on **individual 126-value vectors**:
21 landmarks × 3 coordinates × 2 hands. It does not learn temporal motion
across frames. The new app preserves that input, collects a complete sign,
excludes the final webcam holding period, and averages per-frame probability
evidence. Weak or conflicting evidence is marked as a possible match.

This changes segmentation and evidence collection; it does **not** create an
open-vocabulary translator or claim a new accuracy result. Reliable temporal
recognition needs a separately trained sequence model and an evaluated dataset.
Raw weights contain no normalization or class-name metadata, so the app uses
the original project label order and identity normalization. Scores are not
calibrated probabilities of correctness.

### Watch a supported phrase

Select **Sentences → Explore supported signs**, then choose a phrase. The
**With videos** filter lists the three verified public references: **Xin chào.**,
**Cảm ơn.**, and **Bạn tên là gì?** Playback controls support replay and speed
changes; **Try this sign** returns to the recognition controls. Videos stop
when the dialog closes or you return to the phrase list. Other phrases show
an unavailable state until a verified recording is added.

These learning references may use different regional signs from the training
data. Attribution and instructions for extending the catalog are in
`assets/signs/SOURCES.md`. No third-party video is copied or redistributed.

### Connect word weights

Place your project's trained YOLO detection or classification checkpoint at
`weights/word_model.pt`, then restart the server. Alternatively:

```powershell
$env:VSL_WORD_MODEL = 'C:\path\to\your\word_model.pt'
.\.venv\Scripts\python.exe test.py
```

Its class names become the supported word vocabulary automatically. Letter
and sentence paths can also be overridden with `VSL_LETTER_MODEL` and
`VSL_SENTENCE_MODEL`. Use checkpoints from your own trusted training pipeline.

The old root `model.pt` is an HTML Google Drive viewer page saved with a `.pt`
extension, not PyTorch weights. It is preserved but is not used. Alphabet
weights were downloaded from this project's public repository into
`weights/alphabet_model.pt`.

For a repository clone, default model paths are `../weights/alphabet_model.pt`
and `../weights/sentence_resnet_model.pt`. Environment overrides take priority;
otherwise the app looks for local checkpoints before the parent weights folder.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite exercises real image inference, video decoding, live tracking,
session lifecycle, left/right ordering, corruption/empty-file rejection,
blank-input rejection, natural movement boundaries, jitter, dropped tracking,
and absence of results before the six-second pause. The bundled source-dataset
samples recognize **Y, X, D**. These are pipeline checks, not an independent
accuracy benchmark. Physical camera smoothness still depends on the device,
browser, and lighting.

## Files and sources

- `home/home.html`: introduction and project information.
- `upload/upload.html`, `webcam/webcam.html`: dedicated recognition pages.
- `assets/studio.css` and `studio.js`: shared design and recognition controls.
- `test.py`: FastAPI routes, upload decoding, and session lifecycle.
- `recognition.py`: model adapters, phrase evidence, and capture state machine.
- `predict_realtime.py`: original model architecture and label ordering.
- `assets/samples/SOURCES.md`: unmodified dataset image provenance, CC BY 4.0 attribution.
- `assets/fonts/OFL.txt`: Manrope's SIL Open Font License. Fonts load locally.

Project: https://github.com/MortyPham/Vietnamese-Sign-Language-Recognition-System-Using-Deep-Learning-and-Computer-Vision

Research paper: https://docs.google.com/document/d/1FpAOx5w2dIDbhCpRAPXXTfhFj6V8ickP_8FBTIwYfiU/edit

Training dataset: https://universe.roboflow.com/ho-chi-minh-university-of-technology-clmwp/vietnam-sign-language

Sentence extraction reference: https://github.com/khooinguyeen/Vietnamese-Sign-Language-Translation
