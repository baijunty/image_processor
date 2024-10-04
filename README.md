 # 使用AI处理图片

本项目旨在提供一个基于AI的图像处理工具。以下是如何使用该工具进行图像处理的详细指南。

## 环境准备

确保你已经安装了必要的Python包，可以通过以下命令安装：

```bash
pip install torch transformers
```

## 如何使用

### 初始化ImageProcessor类

首先，你需要实例化`ImageProcessor`类并设置设备（CPU或GPU）。

```python
from image_process import ImageProcessor

processor = ImageProcessor()
```

### 处理图像标签

你可以使用以下方法获取图像的标签：

```python
images = ["path/to/image1.jpg", "path/to/image2.jpg"]  # 替换为你的图像路径
limit = 5  # 你希望返回的标签数量
threshold = 0.5  # 标签置信度阈值

tags = processor.processing_tags(images, limit, threshold)
print(tags)
```

### 提取图像特征

你可以使用以下方法获取图像的特征：

```python
images = ["path/to/image1.jpg", "path/to/image2.jpg"]  # 替换为你的图像路径
features = processor.processing_feature(images)
print(features)
```

### 文本嵌入

你可以使用以下方法获取文本的嵌入：

```python
docs = ["这是第一个文档", "这是第二个文档"]  # 替换为你希望处理的文本
limit = 2  # 你希望返回的相似文本数量
threshold = 0.5  # 嵌入置信度阈值

embeddings = processor.text_embedings(docs, limit, threshold)
print(embeddings)
```

## 常见问题

如果你在处理过程中遇到任何问题，请参考以下资源：

- [PyTorch官方文档](https://pytorch.org/docs/)
- [Transformers库文档](https://huggingface.co/transformers/)

通过以上步骤，你可以使用AI处理图片并获取相关标签、特征和文本嵌入。希望这个指南对你有所帮助！