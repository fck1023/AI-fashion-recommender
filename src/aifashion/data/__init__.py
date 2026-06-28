"""資料層:雙雲版本化 pipeline + 訓練用串流。

  - prepare   : 下載 + 去重/清理/正規化 + 打 WebDataset 分片
  - captions  : heuristic + BLIP captioning、cap_map 覆寫
  - webdataset: 訓練用的 robust WDS 串流(Drive 優先,HF 保底)
  - hub       : 雙雲 artifacts + HF Hub 版本化發佈 + `latest` 別名 + 驗證
"""
