# A 分類訓練程式

請把原本的 `A3_train.py`、`a_common.py`、`a_data_utils.py`（及 Colab_A.ipynb）放在這個資料夾。

訓練設定摘要（取自簡報）：
- 多標籤分類，BCEWithLogits，55 個水果大類；輸入 224×224，ImageNet 平均／標準差
- 階段 1：白底圖 15 epoch、Adam、lr 0.001、批次 64；ResNet18／MobileNetV2 凍結主幹只訓練最後一層
- 階段 2：加入真實照片 10 epoch、lr 0.0001、全部參數可訓練；白底：真實＝1:1；取真實驗證分數最高的 epoch
- 隨機種子 42
