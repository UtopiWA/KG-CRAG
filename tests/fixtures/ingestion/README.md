# 摄取测试 fixture

`sample.pdf` 是只含 ASCII 的可审查微型 PDF，并配有 raw 风格元数据模板。其余测试使用 PyMuPDF 在 pytest 临时目录中由固定字符串生成双栏、加密、空文本和页数上限场景；测试全程离线且字节输入固定。
