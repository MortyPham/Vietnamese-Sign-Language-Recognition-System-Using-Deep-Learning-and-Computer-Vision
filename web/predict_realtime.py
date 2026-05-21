# ------------ CONFIG (edit these if your checkpoint doesn't include them) ------------
CHECKPOINT_PATH = "tinyresnet1d.pt"      # <-- your trained model file
# If your checkpoint doesn't store class names, list them here in the correct order:
CLASS_NAMES_FALLBACK = [
  "ban dang lam gi",
  "ban di dau the",
  "ban hieu ngon ngu ky hieu khong",
  "ban hoc lop may",
  "ban khoe khong",
  "ban muon gio roi",
  "ban phai canh giac",
  "ban ten la gi",
  "ban tien bo day",
  "ban trong cau co the",
  "bo me toi cung la nguoi Diec",
  "cai nay bao nhieu tien",
  "cai nay la cai gi",
  "cam on",
  "cap cuu",
  "chuc mung",
  "chung toi giao tiep voi nhau bang ngon ngu ky hieu",
  "con yeu me",
  "cong viec cua ban la gi",
  "hen gap lai cac ban",
  "mon nay khong ngon",
  "toi bi chong mat",
  "toi bi cuop",
  "toi bi dau dau",
  "toi bi dau hong",
  "toi bi ket xe",
  "toi bi lac",
  "toi bi phan biet doi xu",
  "toi cam thay rat hoi hop",
  "toi cam thay rat vui",
  "toi can an sang",
  "toi can di ve sinh",
  "toi can gap bac si",
  "toi can phien dich",
  "toi can thuoc",
  "toi dang an sang",
  "toi dang buon",
  "toi dang o ben xe",
  "toi dang o cong vien",
  "toi dang phai cach ly",
  "toi dang phan van",
  "toi di sieu thi",
  "toi di toi Ha Noi",
  "toi doc kem",
  "toi khoi benh roi",
  "toi khong dem theo tien",
  "toi khong hieu",
  "toi khong quan tam",
  "toi la hoc sinh",
  "toi la nguoi Diec",
  "toi la tho theu",
  "toi lam viec o cua hang",
  "toi nham dia chi",
  "toi song o Ha Noi",
  "toi thay doi bung",
  "toi thay nho ban",
  "toi thich an mi",
  "toi thich phim truyen",
  "toi viet kem",
  "xin chao",
  "idle"
]   # ====================== CONFIG ======================

import cv2, mediapipe as mp, numpy as np, collections, torch, torch.nn as nn

# ---------- MediaPipe: frame -> 126 ----------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

def mp_frame_to_126(results):
    v = np.zeros(126, dtype=np.float32)
    if not results.multi_hand_landmarks or not results.multi_handedness:
        return v
    hands = {}
    for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = hd.classification[0].label.upper()  # 'LEFT'/'RIGHT'
        hands[label] = lm

    def fill(dst, hand_lm):
        if hand_lm is None: return
        i = 0
        for p in hand_lm.landmark:  # 21 points
            v[dst+i+0] = p.x; v[dst+i+1] = p.y; v[dst+i+2] = p.z
            i += 3

    # Order: LEFT [0..62], RIGHT [63..125]
    fill(0,  hands.get("LEFT",  None))
    fill(63, hands.get("RIGHT", None))
    return v

# ---------- ResNet-1D (for vector length 126) ----------

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, stride=s, padding=k//2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, k, padding=k//2),
            nn.BatchNorm1d(out_ch),
        )
        self.short = nn.Conv1d(in_ch, out_ch, 1, stride=s) if (in_ch!=out_ch or s!=1) else nn.Identity()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.short(x))

class ResNet1D(nn.Module):
    def __init__(self, num_classes, in_ch=1):          # <-- NEW: in_ch parameter (1/2/3)
        super().__init__()
        self.layers = nn.Sequential(
            ResBlock1D(in_ch,  32),                    # <-- use in_ch here
            ResBlock1D(32,     64, s=2),
            ResBlock1D(64,     64),
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layers(x)              # (B, 64, L')
        x = x.mean(dim=2)               # GAP over time
        return self.head(x)             # (B, num_classes)


# ---------- Checkpoint loading ----------
def load_any_checkpoint(path):
    obj = torch.load(path, map_location="cpu")
    # module -> use state_dict
    if isinstance(obj, nn.Module):
        return {"state_dict": obj.state_dict()}
    # dict-like or raw state_dict
    if isinstance(obj, dict):
        return obj
    return {"state_dict": obj}

# ---------- Smoother ----------
class Smooth:
    def __init__(self, n, alpha=0.6):
        self.v = np.zeros(n, dtype=np.float32); self.a = alpha
    def __call__(self, probs):
        self.v = self.a*self.v + (1-self.a)*probs
        return self.v

# ---------- Main demo ----------
def main():
    print("Starting real-time prediction...")
    ckpt = load_any_checkpoint(CHECKPOINT_PATH)
    state = ckpt.get("state_dict", ckpt)
    class_names = ckpt.get("class_names", None) or CLASS_NAMES_FALLBACK
    if not class_names or len(class_names) < 2:
        raise RuntimeError("Please set CLASS_NAMES_FALLBACK to your label list in correct order.")

    n_classes = len(class_names)
    # pick the same hyperparams you used in training
    model = ResNet1D(
        num_classes=n_classes,
        in_ch = 1,
    )
    model.load_state_dict(state, strict=False)
    model.eval()

    # normalization from ckpt if available
    mu = ckpt.get("mu", None); sd = ckpt.get("sd", None)
    if mu is None or sd is None:
        mu = np.zeros(126, dtype=np.float32); sd = np.ones(126, dtype=np.float32)
    else:
        mu = np.asarray(mu, dtype=np.float32); sd = np.asarray(sd, dtype=np.float32) + 1e-8

    softmax = nn.Softmax(dim=-1)
    smooth  = Smooth(n_classes, alpha=0.6)

    # camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # on some PCs CAP_MSMF works better
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Try index 1 or use CAP_MSMF backend.")

    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            v = mp_frame_to_126(results)            # (126,)
            v = (v - mu) / sd
            x = torch.from_numpy(v).view(1, 1, 126).float()

            with torch.no_grad():
                logits = model(x).squeeze(0)        # (C,)
                probs  = softmax(logits).cpu().numpy()
                probs  = smooth(probs)
                top    = int(probs.argmax())
                conf   = float(probs[top])

            # draw landmarks
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"{class_names[top]}  {conf*100:.1f}%",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("ResNet1D – Sign prediction (ESC to quit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
