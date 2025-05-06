# 2025.5

## 2025.5.6

《DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation》

文章要点：

    1. 有一类 image generation 问题，是对图像中 subject 尽量保持不变，**改变其风格/背景等** ，包括 recontextualization, modification of subject properties such as material and species, art rendition, and viewpoint modification 等

    2. 当前 text-to-image model 缺少 preserve the subject's key features and synthesize images of the subject's contextualized in different scenes 的能力，主要原因是，模型output的表达能力受限，不能准确的 reconstruct given subjects 的外观，都只是创建它们的变体

    3. 本文目标，given a few images of a subject，**implant the subject** into the output domain of the model（Pre-trained, diffusion-based text-to-image framework）so that it can be synthesized with a unique identifier（prompt）


![](https://aike0ghfh14.feishu.cn/space/api/box/stream/download/asynccode/?code=Yzc1N2FlNzgxOWNkY2Y3NDYzY2NmNDdiY2VmN2Q0ZTFfRWhpY084M1FPOGEzenk3U1J3aU5YMUYxYWlFaXVuWFhfVG9rZW46RzhPZWJBUkxyb29YRWl4d041OWNWU0FqbkFoXzE3NDY1NDE3NTU6MTc0NjU0NTM1NV9WNA)

![](https://aike0ghfh14.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDMzMTU3MmExZTgyZmZkZWIwZWFjMWU0MDViMjk4MWVfMGxkbGpqaDhKZ040cnkzM0FuSGVBcmFzaExYQjlYbGtfVG9rZW46WUFRT2J6allVb1lpaHF4d2FpWmNma0dCbkJlXzE3NDY1NDE3NjE6MTc0NjU0NTM2MV9WNA)


《Video Probabilistic Diffusion Models in Projected Latent Space》

文章要点：

    1. Video generation 的特点：High resolution, high dimensionality；temporal coherent, temporal dynamic；Large spatial variations

2. 本文的核心想法，是将 given video parameterize 到一个2D latent space中，再训练 diffusion model，以此降低 video 模型训练和推理的复杂度，同时保持比较好的生成效果
3. PVDM（ *projected latent video diffusion model* ）， **the first latent diffusion model designed for video synthesis** , 主要包含2个部分：autoencoder，encode 3D video pixels as three succinct 2D latent vectors，把视频映射到低维空间；Diffusion model

 ![1746512432100](image/papers/1746512432100.png)

## 2025.5.5

《**Retrieval-Augmented Generation with Graphs (GraphRAG)**》

本文要点：

1. 本文是一篇综述性文章，提供一个comprehensive and up-to-date review of GraphRAG to unify the GraphRAG framework from the global perspective
2. RAG 是一个 powerful tech 加强下游任务，通过retrieve additional info，而Graph 是一个包含了massive信息的 nodes 和 edges的集合，**这让它成为一个 golden resource for RAG；**比较缺少 a systematic and up-to-date survey of GraphRAG's key concepts and techniques
3. GraphRAG offers unique advantages in **capturing relational knowledge** by leveraging graph-based machine learning (e.g., Graph Neural Networks (GNNs)) and graph/network analysis techniques (e.g., Graph Traversal  Search and Community Detection)；拥有复杂结构的 graph，需要设计 **dedicated graph encoder** with appropriate expressiveness to capture structural nuances

![1746455991982](image/papers/1746455991982.png)
