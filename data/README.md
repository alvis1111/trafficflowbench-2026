# 数据说明

数据体积约 12GB，不提交到 Git（见 `.gitignore`）。

## 下载

竞赛名 `2026-ieee-big-data-traffic-flow-bench`（Kaggle）。用 Bearer token（`KGAT_` 前缀）：

```bash
kaggle competitions download -c 2026-ieee-big-data-traffic-flow-bench
```

本机数据已下载在 `C:\Users\GEM07\OneDrive\Desktop\kaggle_public`（12GB）。

## 注意事项

- 下载单文件时，路径里的 `/` 要做 `%2F` 编码。
- submission.csv 别用 Excel 打开（会截断到 1048576 行）。
