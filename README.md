# 水果影像辨識系統 Fruit Recognition (Classification + Detection)

**中文** | [English](#english)

從白底資料集到真實照片：A 分類（Baseline CNN／ResNet18／MobileNetV2）＋ B 偵測（YOLOv8s，26 類），整合成 Gradio 介面。

🔗 **線上 Demo**：🔗 線上 Demo：[https://huggingface.co/spaces/PCChen27/Fruit_Recognition_Free_GPU｜　📄 簡報：docs/project_presentation.pdf

## 重點結果（取自專題簡報）
| 項目 | 結果 |
|---|---|
| A 分類（Top1） | ResNet18：白底 98.5%、真實照片 66.0%；MobileNetV2：95.0%／64.9%；Baseline CNN：85.4%／9.3% |
| 只用白底圖訓練 → 加入真實照片 | ResNet18 15.5% → 66.0%；MobileNetV2 19.6% → 64.9%（真實照片 Top1） |
| B 偵測（26 類，資料集 test） | mAP@0.5 0.772、mAP@0.5:0.95 0.501、Precision 74.0%、Recall 75.9% |
| 真實與圖庫照片（範圍內 150 張）框到正確種類 | 舊 B 64.0% → 新 B 72.0% |
| A、B 至少一個答對（單一水果 176 張） | 71.0% → 76.7% |

**取捨**：新 B 的錯誤框從 640 增加到 934，框對率 57.1% → 54.7%；新舊測試組不同，資料集成績不能直接比較。詳見簡報第 5～7 頁。

## 兩種模式
- **一般模式**：固定 R0（信心 0.35、重疊 0.5、640 px）；完全沒框到時自動以信心 0.25 重試並提示較不確定。
- **研究模式**：信心、重疊、圖片大小、A 輔助重新命名（R9 方式）、2×2 分塊可調，可套用 R0／R1／R3／R9；手動調整標示為「自訂」。R0～R9 完整紀錄見 [experiments/R0_R9.md](experiments/R0_R9.md)。

R9 最保守，但仍會改壞少數原本正確的結果、速度較慢，所以只放研究模式。

## 執行
```bash
pip install -r requirements.txt
# 放入權重：models/classification/{resnet18_final.pt, mobilenetv2_final.pt, classes.json}、models/detection/best.pt
python app/app.py          # APP_MODE=normal 或 research 可只開單一模式
```
權重很大時用 Git LFS（已含 `.gitattributes`）或放在 Hugging Face Model repo，並設定環境變數 `HF_MODEL_REPO=<帳號>/<模型庫>`（repo 內路徑：`classification/…`、`detection/best.pt`）。
A 類別與 B 類別名稱不同時（例如 limes／Lime），複製 `models/name_map.example.json` 為 `name_map.json` 補對照。

## 專案結構
```
app/app.py            Gradio 介面（一般／研究模式）
src/                  A、B 訓練程式
models/               權重（LFS 或 HF）
experiments/          R0～R9 實驗紀錄
results/ screenshots/ docs/
```

## 資料來源與授權（公開前請再確認）
- Fruits-360（Mureşan & Oltean），CC BY-SA 4.0：使用需標註出處，改作需以相同條款分享。
- Multi-Class Fruit and Vegetable Detection Dataset（George Bayón & Sánchez González，Zenodo 10.5281/zenodo.18904638），CC BY 4.0。
- 補充資料：LVIS、Jackfruit、cantaloupe 等，請依各自條款標註。
- 測試圖庫照片：Unsplash、Pexels，攝影師名單見 stock_test_set.csv；請勿整包重新散布原圖。
- 用上述資料訓練的權重，公開前請確認各資料集的條款與標註要求。

## 限制
範圍外水果（如油桃、龍眼）不在 B 類別內；測試照片小量且手選，不代表一般情況；名稱對錯由程式判定，換用前建議抽查（尤其覆盆子）。
