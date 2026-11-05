---
title: Fruit Recognition
emoji: 🍎
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 水果影像辨識系統

上傳一張照片：**A** 判斷是什麼水果，**B**（YOLOv8s，26 類）框出位置與數量。

- **一般模式**：直接辨識（固定 R0 設定）。
- **研究模式**：調整信心／重疊／圖片大小，或套用 R0、R1、R3、R9 實驗設定。R9 為研究用途。

⏱ 免費 CPU 上一張照片約 15 秒；960 px 與分塊更慢，建議先試範例圖。
原始碼與完整實驗紀錄：https://github.com/<你的帳號>/Fruit-Recognition

資料：Fruits-360（CC BY-SA 4.0）、Zenodo 水果偵測資料集（CC BY 4.0）。模型僅供展示。

## 部署
把 `app/app.py`（改名為 `app.py` 放在 Space 根目錄，路徑會自動判斷）、`requirements.txt`、`models/`（或設定 Space 變數 `HF_MODEL_REPO`）、`examples/` 放到 Space。
