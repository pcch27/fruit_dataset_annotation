"""B：YOLOv8s 訓練（與 v12 簡報記載的設定一致）。data.yaml 請指向 26 類水果資料集。"""
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(
    data="data.yaml",   # 26 類：21 類水果＋奇異果、覆盆子、黑莓、哈密瓜、菠蘿蜜
    epochs=50, patience=15, batch=16, imgsz=640, seed=0, amp=True,
)
metrics = model.val(data="data.yaml", split="test", imgsz=640)
print("mAP50", metrics.box.map50, "mAP50-95", metrics.box.map)
