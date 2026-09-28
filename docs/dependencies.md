# 关键依赖说明

## PyMuPDF

- 固定版本：`pymupdf==1.28.2`。
- 用途：读取本地论文 PDF，提取页面、文本块、字体和边界框信息。
- Python 兼容范围：本项目使用 Python 3.11；该版本要求 Python 3.10 或更高版本。
- 许可证：PyMuPDF 以 GNU AGPL v3 和商业许可证双重授权。发布或部署本项目时，必须依据实际分发方式评估并履行相应许可证义务；若不能接受 AGPL 条款，应先取得商业许可证或更换解析器。
- 上游资料：[PyPI 项目页](https://pypi.org/project/PyMuPDF/)、[官方许可说明](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)。

解析器通过 `import pymupdf` 导入，不使用已弃用的 `fitz` 顶层名称。依赖版本只能在 OpenSpec 变更中调整，并需要重新运行离线解析测试和真实试点质量门禁。
