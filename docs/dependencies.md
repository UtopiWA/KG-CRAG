# 关键依赖说明

## PyMuPDF

- 固定版本：`pymupdf==1.28.2`。
- 用途：读取本地论文 PDF，提取页面、文本块、字体和边界框信息。
- Python 兼容范围：本项目使用 Python 3.11；该版本要求 Python 3.10 或更高版本。
- 许可证：PyMuPDF 以 GNU AGPL v3 和商业许可证双重授权。发布或部署本项目时，必须依据实际分发方式评估并履行相应许可证义务；若不能接受 AGPL 条款，应先取得商业许可证或更换解析器。
- 上游资料：[PyPI 项目页](https://pypi.org/project/PyMuPDF/)、[官方许可说明](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)。

解析器通过 `import pymupdf` 导入，不使用已弃用的 `fitz` 顶层名称。依赖版本只能在 OpenSpec 变更中调整，并需要重新运行离线解析测试和真实试点质量门禁。

## Dense RAG 依赖

- `qdrant-client==1.13.2`：连接 Compose 中固定的 Qdrant `v1.13.2`，采用 Apache-2.0 许可证。客户端只出现在 `vector_store` 适配器层；默认测试使用内存实现。
- `sentence-transformers==3.4.1`：加载本地 Embedding 模型，采用 Apache-2.0 许可证。其 PyTorch、Transformers 等传递依赖必须具有 Python 3.11/3.12 可用 wheel；默认测试通过注入替身模型避免下载权重。
- `openai==1.63.2`：实现 OpenAI-compatible LLM Provider，采用 Apache-2.0 许可证。SDK 只出现在 `providers` 适配器层，凭据仅从环境变量读取。

默认 Embedding 模型为 `BAAI/bge-m3`，配置固定修订 `5617a9f61b028005a4858fdac845db406aefb181`，模型许可证为 MIT。首次显式运行真实索引命令时可能需要从模型仓库下载权重；导入包、运行默认 pytest 或 dry-run 不得触发下载。生产或课程交付前应同时复核模型卡、上游依赖许可证和实际分发方式。
