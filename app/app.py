"""水果影像辨識系統：A 分類（ResNet18／MobileNetV2）＋ B 偵測（YOLOv8s，26 類）

一般模式：固定 R0，沒框到水果時自動以信心 0.25 重試（等同 R1）。
研究模式：信心／重疊／圖片大小／A 輔助重新命名（R9）／分塊辨識可調，可套用 R0、R1、R3、R9。

啟動：
    python app/app.py                 # 同時顯示兩個分頁
    APP_MODE=normal python app/app.py # 只開一般模式
    APP_MODE=research python app/app.py
"""
from __future__ import annotations

import json
import os
import time
import zlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
ROOT = _HERE if (_HERE / "models").exists() else _HERE.parent  # HF Space 把 app.py 放根目錄時也能用
A_DIR = ROOT / "models" / "classification"
B_WEIGHTS = ROOT / "models" / "detection" / "best.pt"
EXAMPLES = ROOT / "examples"

# 若本機沒有權重，且設定了 HF_MODEL_REPO（例如 "帳號/fruit-recognition-models"），就從 Hugging Face 下載
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")
A_ARCHS = {"ResNet18": "resnet18_final.pt", "MobileNetV2": "mobilenetv2_final.pt"}
APP_MODE = os.environ.get("APP_MODE", "both")  # both | normal | research


# --------------------------------------------------------------------------
# 設定與預設組合（R0～R9 中放進介面的幾組）
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    conf: float = 0.35          # 信心門檻
    iou: float = 0.5            # 重疊框合併門檻
    imgsz: int = 640            # 圖片大小
    relabel: bool = False       # A 輔助重新命名（R9 方式）
    tiling: bool = False        # 自動 2×2 分塊
    agnostic: bool = True       # 不同種類的重疊框也合併（v12 預設）
    min_box: float = 0.001      # 過濾小於整張圖此比例的框（v12 預設 0.001）


PRESETS = {
    "R0 基準（信心 0.35、重疊 0.5、640 px）": Params(),
    "R1 信心 0.25": Params(conf=0.25),
    "R3 信心 0.25＋960 px": Params(conf=0.25, imgsz=960),
    "R9 A 輔助重新命名（保守版）": Params(relabel=True),
}
PRESET_NOTES = {
    "R0": "基準，一般模式使用。",
    "R1": "較寬鬆，只救回少數沒框到的照片。",
    "R3": "救回較多沒框到的照片，但常框成錯誤種類，錯誤框會增加。",
    "R9": "最保守的 A 輔助方式，但仍會改壞少數原本正確的結果，且每個框多跑一次 A，速度較慢。",
}
RETRY_CONF = 0.25
R9_CONF = 0.9          # B 原有類別，A 信心達此值才改名
TILE_MIN_BOXES = 8     # 框到幾個以上才分塊
TILE_MIN_SIDE = 1000   # 長邊至少幾 px 才分塊


def label_for(p: Params) -> str:
    """依參數反推名稱；與預設完全相同才叫 R0/R1/R3/R9，否則標示為自訂。"""
    for name, preset in PRESETS.items():
        if preset == p:
            return name.split()[0]
    return "自訂"


# --------------------------------------------------------------------------
# 模型載入（延遲載入，只載一次）
# --------------------------------------------------------------------------
_cache: dict = {}


def _fetch(local: Path, repo_path: str) -> Path:
    if local.exists():
        return local
    if HF_MODEL_REPO:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(HF_MODEL_REPO, repo_path))
    raise FileNotFoundError(f"找不到模型檔：{local}。請放入權重，或設定環境變數 HF_MODEL_REPO。")


def load_detector():
    if "yolo" not in _cache:
        from ultralytics import YOLO

        _cache["yolo"] = YOLO(str(_fetch(B_WEIGHTS, "detection/best.pt")))
    return _cache["yolo"]


def load_classifier(arch: str):
    key = f"a_{arch}"
    if key not in _cache:
        import torch
        import torch.nn as nn
        from torchvision import models

        classes = json.loads(_fetch(A_DIR / "classes.json", "classification/classes.json").read_text(encoding="utf-8"))
        if arch == "ResNet18":
            m = models.resnet18(weights=None)
            m.fc = nn.Linear(m.fc.in_features, len(classes))
        else:
            m = models.mobilenet_v2(weights=None)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(classes))
        path = _fetch(A_DIR / A_ARCHS[arch], f"classification/{A_ARCHS[arch]}")
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model", state))
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        m.load_state_dict(state)
        m.eval()
        _cache[key] = (m, classes)
    return _cache[key]


def name_map() -> dict[str, str]:
    """A 類別名稱 → B 類別名稱（小寫比對）。可在 models/name_map.json 補別名，如 {"lemon": "Lemon"}。"""
    p = ROOT / "models" / "name_map.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# --------------------------------------------------------------------------
# A：分類（多標籤，sigmoid）
# --------------------------------------------------------------------------
def classify(img: Image.Image, arch: str = "ResNet18", top_k: int = 5):
    import torch
    from torchvision import transforms

    model, classes = load_classifier(arch)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        probs = torch.sigmoid(model(tf(img.convert("RGB")).unsqueeze(0)))[0].numpy()
    order = np.argsort(-probs)[:top_k]
    return [(classes[i], float(probs[i])) for i in order]


# --------------------------------------------------------------------------
# B：偵測（含分塊）
# --------------------------------------------------------------------------
def _detect_once(img: Image.Image, p: Params, conf: float):
    model = load_detector()
    r = model.predict(np.array(img.convert("RGB")), conf=conf, iou=p.iou, imgsz=p.imgsz,
                      agnostic_nms=p.agnostic, verbose=False)[0]
    b = r.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int), model.names
    return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int), model.names


def _filter_small(xyxy, conf, cls, w, h, min_ratio):
    if len(xyxy) == 0:
        return xyxy, conf, cls
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) / float(w * h)
    keep = area >= min_ratio
    return xyxy[keep], conf[keep], cls[keep]


def merge_boxes(xyxy, conf, cls, iou_thr):
    """合併分塊與整張的框（不分種類 NMS）。"""
    if len(xyxy) == 0:
        return xyxy, conf, cls
    import torch
    from torchvision.ops import nms

    keep = nms(torch.tensor(xyxy, dtype=torch.float32), torch.tensor(conf, dtype=torch.float32), iou_thr).numpy()
    return xyxy[keep], conf[keep], cls[keep]


def detect(img: Image.Image, p: Params, conf: float | None = None):
    conf = p.conf if conf is None else conf
    w, h = img.size
    xyxy, cf, cl, names = _detect_once(img, p, conf)
    # 分塊：整張框到 8 個以上且長邊 ≥1000 px，才切 2×2 再辨識（v12 規則）
    if p.tiling and len(xyxy) >= TILE_MIN_BOXES and max(w, h) >= TILE_MIN_SIDE:
        parts = [(xyxy, cf, cl)]
        for ix in range(2):
            for iy in range(2):
                box = (ix * w // 2, iy * h // 2, (ix + 1) * w // 2, (iy + 1) * h // 2)
                tx, tc, tl, _ = _detect_once(img.crop(box), p, conf)
                if len(tx):
                    tx = tx + np.array([box[0], box[1], box[0], box[1]])
                parts.append((tx, tc, tl))
        xyxy = np.concatenate([q[0] for q in parts])
        cf = np.concatenate([q[1] for q in parts])
        cl = np.concatenate([q[2] for q in parts])
        xyxy, cf, cl = merge_boxes(xyxy, cf, cl, p.iou)
    xyxy, cf, cl = _filter_small(xyxy, cf, cl, w, h, p.min_box)
    labels = [names[int(c)] for c in cl]
    return xyxy, cf, labels, set(names.values())


def relabel_r9(img: Image.Image, xyxy, labels, b_names: set[str], arch: str = "ResNet18"):
    """R9：A 只補 B 沒有的名稱；B 原有的類別要 A 信心 ≥0.9 才改。"""
    nm = {k.lower(): v for k, v in name_map().items()}
    b_lower = {n.lower() for n in b_names}
    out, changed = [], 0
    for (x1, y1, x2, y2), lab in zip(xyxy, labels):
        crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
        if min(crop.size) < 8:
            out.append(lab)
            continue
        a_name, a_conf = classify(crop, arch, top_k=1)[0]
        a_as_b = nm.get(a_name.lower(), a_name)
        if a_as_b.lower() == lab.lower():
            out.append(lab)
        elif a_as_b.lower() not in b_lower:      # B 沒有這個名稱 → 補上
            out.append(a_as_b)
            changed += 1
        elif a_conf >= R9_CONF:                   # B 有，但 A 非常有把握
            out.append(a_as_b)
            changed += 1
        else:
            out.append(lab)
    return out, changed


# --------------------------------------------------------------------------
# 繪圖與結果整理
# --------------------------------------------------------------------------
def _color(name: str):
    h = zlib.crc32(name.encode()) % (256 ** 3)
    return (60 + h % 160, 60 + (h >> 8) % 160, 60 + (h >> 16) % 160)


def draw(img: Image.Image, xyxy, conf, labels) -> Image.Image:
    from PIL import ImageDraw

    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    lw = max(2, min(out.size) // 200)
    for (x1, y1, x2, y2), c, lab in zip(xyxy, conf, labels):
        col = _color(lab)
        d.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        d.text((x1 + 3, max(0, y1 - 12)), f"{lab} {c:.2f}", fill=col)
    return out


def summarize(labels) -> str:
    if not labels:
        return "沒有偵測到水果。"
    counts: dict[str, int] = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return "、".join(f"{k} × {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])) + f"（共 {len(labels)} 個框）"


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def run(img, p: Params, auto_retry: bool, arch: str = "ResNet18"):
    t0 = time.time()
    if img is None:
        return None, "請先上傳照片。", [], ""
    img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    msgs = []
    xyxy, cf, labels, b_names = detect(img, p)
    if auto_retry and len(xyxy) == 0:
        xyxy, cf, labels, b_names = detect(img, p, conf=RETRY_CONF)
        if len(xyxy):
            msgs.append(f"⚠️ 預設設定沒框到水果，已改用信心 {RETRY_CONF} 重試；結果較不確定。")
    if p.relabel and len(xyxy):
        labels, changed = relabel_r9(img, xyxy, labels, b_names, arch)
        msgs.append(f"A 輔助重新命名：{changed} 個框被改名。")
    a_top = classify(img, arch, top_k=5)
    if len(xyxy) == 0:
        msgs.append(f"B 沒有框到水果；A 的第一名判斷是 {a_top[0][0]}（{a_top[0][1]:.0%}）。")
    drawn = draw(img, xyxy, cf, labels)
    table = [[n, f"{s:.1%}"] for n, s in a_top]
    info = (f"**B 偵測**：{summarize(labels)}  \n"
            f"**設定**：{label_for(p)}｜信心 {p.conf}｜重疊 {p.iou}｜{p.imgsz} px｜"
            f"A 重新命名 {'開' if p.relabel else '關'}｜分塊 {'開' if p.tiling else '關'}｜"
            f"用時 {time.time() - t0:.1f} 秒\n\n" + "  \n".join(msgs))
    return drawn, info, table, ""


def run_normal(img, arch):
    return run(img, PRESETS[next(iter(PRESETS))], auto_retry=True, arch=arch)[:3]


def run_research(img, conf, iou, imgsz, relabel, tiling, agnostic, arch):
    p = Params(conf=float(conf), iou=float(iou), imgsz=int(imgsz), relabel=bool(relabel), tiling=bool(tiling), agnostic=bool(agnostic))
    return run(img, p, auto_retry=False, arch=arch)[:3]


def apply_preset(name):
    p = PRESETS[name]
    key = name.split()[0]
    return p.conf, p.iou, p.imgsz, p.relabel, p.tiling, p.agnostic, f"**{key}**：{PRESET_NOTES.get(key, '')}"


# --------------------------------------------------------------------------
# 介面
# --------------------------------------------------------------------------
def build_ui():
    import gradio as gr

    ex = [str(p) for p in sorted(EXAMPLES.glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}] if EXAMPLES.exists() else []
    arch_choices = list(A_ARCHS)
    head = ("# 🍎 水果影像辨識系統\n"
            "A 分類判斷「是什麼水果」，B 偵測框出「位置與數量」。免費 CPU 上一張照片約需 15 秒，請稍候；"
            "研究模式的 960 px 與分塊會更慢。")
    with gr.Blocks(title="水果影像辨識", theme=gr.themes.Soft()) as demo:
        gr.Markdown(head)
        with gr.Tabs():
            if APP_MODE in ("both", "normal"):
                with gr.Tab("一般模式"):
                    with gr.Row():
                        with gr.Column():
                            img_n = gr.Image(type="pil", label="上傳照片")
                            arch_n = gr.Dropdown(arch_choices, value=arch_choices[0], label="A 分類模型")
                            btn_n = gr.Button("辨識", variant="primary")
                            if ex:
                                gr.Examples(ex, inputs=img_n)
                        with gr.Column():
                            out_n = gr.Image(label="B：偵測結果")
                            info_n = gr.Markdown()
                            tab_n = gr.Dataframe(headers=["A：水果", "分數"], label="A：分類（前 5 名）")
                    btn_n.click(run_normal, [img_n, arch_n], [out_n, info_n, tab_n])
            if APP_MODE in ("both", "research"):
                with gr.Tab("研究模式"):
                    with gr.Row():
                        with gr.Column():
                            img_r = gr.Image(type="pil", label="上傳照片")
                            preset = gr.Dropdown(list(PRESETS), value=next(iter(PRESETS)), label="套用實驗設定")
                            note = gr.Markdown(f"**R0**：{PRESET_NOTES['R0']}")
                            conf = gr.Dropdown([0.25, 0.35, 0.5, 0.7], value=0.35, label="信心門檻")
                            iou = gr.Dropdown([0.3, 0.5], value=0.5, label="重疊框門檻")
                            imgsz = gr.Dropdown([640, 960], value=640, label="圖片大小（px，960 較慢）")
                            relabel = gr.Checkbox(False, label="A 輔助重新命名（R9 方式，較慢）")
                            tiling = gr.Checkbox(False, label="自動 2×2 分塊辨識（較慢）")
                            agn = gr.Checkbox(True, label="不同種類的重疊框也合併")
                            arch_r = gr.Dropdown(arch_choices, value=arch_choices[0], label="A 分類模型")
                            btn_r = gr.Button("辨識", variant="primary")
                            gr.Markdown("手動調整參數後，結果會標示為「自訂」，不會稱為 R1 或 R9。自動重試在此模式關閉。")
                            if ex:
                                gr.Examples(ex, inputs=img_r)
                        with gr.Column():
                            out_r = gr.Image(label="B：偵測結果")
                            info_r = gr.Markdown()
                            tab_r = gr.Dataframe(headers=["A：水果", "分數"], label="A：分類（前 5 名）")
                    preset.change(apply_preset, preset, [conf, iou, imgsz, relabel, tiling, agn, note])
                    btn_r.click(run_research, [img_r, conf, iou, imgsz, relabel, tiling, agn, arch_r], [out_r, info_r, tab_r])
        gr.Markdown("資料來源：Fruits-360（CC BY-SA 4.0）、Zenodo 水果偵測資料集（CC BY 4.0）。模型僅供展示，結果可能有誤。")
    return demo


if __name__ == "__main__":
    build_ui().queue(max_size=8).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
