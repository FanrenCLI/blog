---
title: RAG
date: 2026-5-30 17:23:00
cover: true
top: true
categories:
  - AI
  - RAG
tags:
  - RAG
author: Fanrencli
---

# RAG

RAG(Retrieval-Augmented Generation)检索增强生成，是一种基于检索和生成的混合模型，用于生成符合用户需求的文本。其核心原理是利用外部知识库或文档集合为大模型提供实时且准确的背景信息，从而弥补大模型的局限性。
RAG模型通常由检索模块和生成模块组成，检索模块用于从数据库中检索与用户查询相关的文本，生成模块用于根据检索到的文本生成符合用户需求的文本。

## RAG 的演进范式

按照技术演进，RAG 大致可以分为三代：

| 阶段 | 核心链路 | 特征 | 典型问题 |
| --- | --- | --- | --- |
| Naive RAG（朴素 RAG） | 索引 → 一次 Top-K 检索 → 拼接上下文 → 生成 | 实现简单，一条直线流水线 | 检索质量不可控、噪声上下文多、专有名词命中率低、无法处理多跳推理 |
| Advanced RAG（高级 RAG） | 在检索前 / 检索中 / 检索后增加处理环节 | 查询改写、混合检索、Rerank 重排序、上下文压缩 | 链路变长，需要模块间联合调优 |
| Modular RAG（模块化 RAG） | 各能力拆成可编排模块，由路由或 Agent 动态决策 | 自适应检索、循环检索、工具调用、长期记忆 | 架构复杂，成本与延迟更高 |

一个完整的高级 RAG 系统全链路如下：

```
离线阶段：原始文档 → 加载解析 → 清洗 → 分块(Chunking) → 元数据增强 → Embedding 向量化 → 写入向量库 / 倒排索引
在线阶段：用户问题 → 查询预处理(改写 / 扩展 / 路由) → 多路召回(向量 + BM25) → 融合(RRF)
         → Rerank 精排 → 上下文压缩 / 过滤 → Prompt 组装 → LLM 生成(带引用) → 评估与监控
```

后面的内容就按照这条链路展开。

## 检索模块

检索模块通常由向量数据库、向量检索器、向量相似度计算器等组成。向量数据库用于存储和管理向量表示，向量检索器用于根据用户查询的向量表示从数据库中检索与之最相关的向量表示，向量相似度计算器用于计算用户查询的向量表示与其他向量表示的相似度。

![检索模块结构](http://fanrencli.cn/fanrencli.cn/rag1.png)

向量数据库构建流程，通常包括以下步骤：
1. 数据预处理：将原始文本数据进行分词（一般）、去停用词、词干提取等处理。
2. 向量表示构建：使用向量表示模型（如词嵌入、向量化等）对预处理后的文本进行向量化，将文本转换为向量表示，以下给出主流的文档转向量的模型：

| 模型 | 类型 | 向量维度 | 最大Token | MTEB得分 | 价格 (每百万token) | 核心优势 | 适用场景 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-Embedding-8B | 开源 | 4096 | 32768 | 70.58 | 免费 | 中文最强，多语言SOTA | 中文/多语言知识库，追求极致召回率 |
| OpenAI text-embedding-3-small | 闭源(API) | 1536 | 8192 | 62.3 | $0.02 | 性价比之王，接入简单稳定 | 90%的通用RAG场景，快速验证原型 |
| OpenAI text-embedding-3-large | 闭源(API) | 3072 | 8192 | 64.6 | $0.13 | 精度标杆，企业级可靠 | 对精度要求极高的生产环境 |
| BGE-M3 | 开源 | 1024 | 8192 | 63.0 | 免费 | 开源全能，支持100+语言和混合检索 | 多语言、长文档场景，作为开源baseline |
| Cohere Embed v4 | 闭源(API) | 1536 | 128K | 65.2 | $0.12 | 超长上下文，支持多模态(文本+图片) | 处理整篇论文、合同等超长文档 |
| Jina v5 text | 开源 | - | - | - | 免费 | 轻量部署，资源占用少 | 资源受限或对延迟敏感的场景 |

3. 向量数据库构建：将构建好的向量表示存储到向量数据库中，将文本转换为向量表示的映射关系存储到向量数据库中。以下给出主流的向量数据库：

| 数据库 | 类型 | 开源协议 | 部署方式 | 核心优势 | 适合场景 |
| --- | --- | --- | --- | --- | --- |
| Chroma | 独立向量库 | Apache 2.0 | 自托管/嵌入式 | 上手极快，与LangChain/LlamaIndex深度集成 | 原型验证、本地开发，数据量小（<100万） |
| Qdrant | 独立向量库 | Apache 2.0 | 自托管/云服务 | 高性能（Rust编写），资源占用低，支持高级过滤 | 追求性价比的中小型生产项目，对性能和过滤要求高 |
| Milvus | 独立向量库 | Apache 2.0 | 自托管/云服务 | 功能最全，支持十亿级向量，CNCF毕业项目，生态完善 | 大规模（千万级以上）、私有化部署，有较强技术运维团队 |
| Pinecone | 独立向量库 | 闭源 | 全托管SaaS | 零运维，查询延迟稳定，按量付费 | 快速上线、不想管理基础设施的原型或生产项目 |
| pgvector | 融合型（PG扩展） | PostgreSQL协议 | 自托管 | 与PostgreSQL无缝集成，可用SQL操作，架构简单 | 已有PostgreSQL，且向量与关系数据强关联的中小型项目 |
| Weaviate | 独立向量库 | BSD-3 | 自托管/云服务 | 内置混合检索和Rerank模块，开箱即用 | 需要混合检索（语义+关键词）的生产场景 |

4. 向量检索器构建：根据用户查询的向量表示，从向量数据库中检索与之最相关的向量表示。当前向量数据库中较为常用的检索算法有：
- 点积检索（Dot Product Retrieval）：根据查询向量与数据库中向量的点积，返回点积最大的向量。
- 距离检索（Distance Retrieval）：根据查询向量与数据库中向量的距离，返回距离最小的向量。
- 倒排索引（Inverted Index）：根据查询向量的特征，返回包含该特征的向量。倒排索引的原理是将向量表示中的每个特征（如词频、词向量等）作为索引，将包含该特征的向量存储在索引中。通过检索索引，可以快速定位到包含查询向量特征的向量，从而提高检索效率。


以上我们介绍了检索模块的构建流程，其中值得注意的是：
- 文本转向量，我们通过直接调用了已有的模型进行向量化，但是这个部分也可以当作一个知识点进行扩展学习，达到自己可以实现文本转向量的功能。
- 向量数据库本文只作为一个工具使用，其中如何检索（向量排序，倒排索引）等知识点也需要了解。

### 文档分块策略（Chunking）

分块是 RAG 效果的"地基"，它决定了一条向量承载多少语义：

- 块太大：一个块内主题混杂，向量被多种语义"平均"，检索不聚焦，还会挤占有限的上下文窗口；
- 块太小：语义残缺，比如检索到"它的取值为 3.14"却不知道"它"指什么，召回了也无法支撑回答。

经验起点是 `chunk_size = 256~512 tokens`，`overlap` 取块大小的 10%~20%（重叠是为了避免关键信息恰好被切断在两个块的边界）。FAQ、问答类语料可以更小，报告、合同类文档可以更大，最终取值一定要在自己的评估集上对比确定。

常见分块策略：

| 策略 | 做法 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 固定长度分块 | 按 token / 字符数滑窗硬切 + overlap（本文代码示例即此方式） | 简单、块大小均匀、便于批量处理 | 可能从句子中间、标题处切断 |
| 递归字符分块 | 按"段落 `\n\n` → 换行 → 句号 → 空格"的优先级递归切分，超限时才降级到下一级分隔符 | 尽量保留自然语义边界，通用首选（LangChain `RecursiveCharacterTextSplitter`） | 仍依赖文档格式，解析糟糕的 PDF 效果打折 |
| 结构化分块 | 按 Markdown 标题、HTML 标签、文档目录切分，并保留标题路径（如"第3章 > 3.2 安装"）；代码按函数 / 类（AST）切 | 元数据丰富，可按章节过滤与溯源 | 依赖文档结构质量 |
| 语义分块 | 计算相邻句子 embedding 的相似度，在话题发生明显变化的"断点"处切分 | 块内主题高度一致 | 计算成本高，相似度阈值难调 |
| 父子分块（Small-to-Big） | 用小块做检索保证精确命中，命中后返回它所属的大块 / 父块喂给 LLM | 兼顾检索精度与上下文完整性 | 实现稍复杂，需维护父子映射 |
| 命题分块（Proposition） | 用 LLM 把段落改写为多条独立、自包含的事实命题后再入库 | 检索精度极高 | 预处理慢、调用成本高 |

分块时建议同时保存元数据：`source`（文件 / URL）、标题、章节路径、页码、`chunk_index`、更新时间。后面的元数据过滤、引用溯源、增量更新都依赖它。

另外要注意：前文数据预处理中提到的"分词、去停用词、词干提取"是 BM25 / TF-IDF 时代的处理方式；使用预训练 embedding 模型时一般**不需要**人工做这些（模型内部自带子词切分），但清洗掉页眉页脚、导航栏、版权声明这类噪声依然很有必要。

### 相似度度量与 ANN 索引

前面提到的"点积 / 距离"本质上是**相似度度量**；而真正支撑十亿级向量秒级检索的，是 **ANN（Approximate Nearest Neighbor，近似最近邻）索引算法**。两者处于不同层面，经常被混为一谈。

**相似度度量：**

| 度量 | 关注点 | 说明 |
| --- | --- | --- |
| 余弦相似度（Cosine） | 向量夹角（方向） | 对模长不敏感，文本检索的默认选择；embedding 归一化后与点积等价 |
| 点积（Dot Product） | 方向 + 模长 | 向量未归一化时，模长大的文档（通常更长 / 更高频）天然得分高 |
| 欧氏距离（L2） | 空间直线距离 | 越小越相似，图像、坐标等场景更常用 |

度量方式必须与 embedding 模型的训练方式、建库时的索引配置保持一致（例如代码里配置的 `hnsw:space: cosine`），否则结果没有意义。

**主流 ANN 索引：**

| 索引 | 原理 | 优点 | 缺点 |
| --- | --- | --- | --- |
| Flat（暴力检索） | 对库中所有向量逐条精确计算 | 100% 召回，适合作为评测基线 | 复杂度 O(N)，百万级以上不可接受 |
| IVF（倒排文件索引） | k-means 把向量空间聚成 `nlist` 个桶，查询时只扫描最近的 `nprobe` 个桶 | 简单有效，用 `nprobe` 调节速度 / 召回 | 查询点位于簇边界时可能漏桶 |
| HNSW（分层可导航小世界图） | 构建多层图，顶层稀疏、底层稠密，查询从顶层快速向下层导航逼近目标 | 查询快、召回高，当前的事实标准（Chroma / Qdrant / pgvector 默认） | 内存占用大；关键参数为 `M`（连接数）、`efConstruction`、`efSearch` |
| PQ（乘积量化） | 把高维向量切成若干子段分别聚类压缩，用聚类编号代替原始浮点值 | 内存骤降，常与 IVF 组合为 IVFPQ | 有损压缩，精度下降 |
| LSH（局部敏感哈希） | 相似向量以高概率落入同一个哈希桶 | 方法经典、有理论保证 | 超高维场景下效果一般 |
| 量化 + 磁盘索引 | 标量 / 二值量化，热数据放内存、冷数据放磁盘（DiskANN 路线） | 超大规模下成本极低 | 工程实现复杂 |

另外需要澄清：前面提到的**倒排索引其实是关键词检索（TF-IDF / BM25）的基础**——它维护的是"词 → 包含该词的文档列表"的映射，而不是直接对向量建索引。它和向量检索不是替代关系，而是互补关系，这正是下一节混合检索的由来。

### 高级检索技术

#### 1. 混合检索（Hybrid Search）

- 稠密向量检索（Dense）：理解语义，"怎么重启电脑"可以匹配到"怎样重新启动计算机"，但对精确字符串（型号、错误码）不敏感；
- 稀疏关键词检索（Sparse，BM25）：基于倒排索引做关键词精确匹配，在专有名词、人名、型号、错误代码、代码标识符（如 `StatusCode 4021`）上的命中率远高于向量。

工业界的常规做法是**两路各召回 Top-K，再用 RRF（Reciprocal Rank Fusion，倒数排名融合）合并排序**：

`RRF(d) = Σ 1 / (k + rank_r(d))`

其中 `rank` 从 1 开始计数，`k` 通常取 60。RRF 只依赖排名而不依赖原始分数，天然规避了 BM25 分数与余弦相似度分数量纲不同、无法直接相加的问题。Qdrant、Weaviate、Milvus、pgvector 都原生支持混合检索。

#### 2. Rerank 重排序（两阶段检索）

1. 粗排：用向量 ANN 快速召回 20~100 个候选，宁多勿漏；
2. 精排：用 Cross-Encoder 把 `query + 候选块` 拼接后联合编码打分，只保留 Top 3~5 喂给 LLM。

普通 embedding 模型是**双塔结构（Bi-Encoder）**：查询和文档各自独立编码，速度快但两者在编码时没有任何信息交互；Cross-Encoder 中 query 与文档的注意力可以双向流动，判断准得多，但每个候选都要算一次，无法预先建库，所以只能放在候选数已经很少的精排阶段。

常用 Rerank 模型：开源的 `BAAI/bge-reranker-v2-m3`、`Qwen3-Reranker`、`jina-reranker-v2`，闭源的 Cohere Rerank 3.5。

#### 3. 查询预处理（Pre-retrieval）

| 技术 | 做法 | 解决的问题 |
| --- | --- | --- |
| Query Rewriting | 结合对话历史把指代表述补全（"它多少钱" → "iPhone 16 的价格"） | 多轮对话中 query 语义不完整 |
| Multi-Query | 让 LLM 从不同角度生成 3~5 条改写，分别检索后合并去重 | 用户措辞与文档措辞不一致 |
| HyDE | 先让 LLM 生成一个"假设性答案"（哪怕包含事实错误），用它的向量去检索 | 问题过短、与文档语言风格差距大；假设答案与真实文档措辞更接近 |
| Step-back Prompting | 把具体问题抽象一层（"该药 2023 年在法国获批了吗" → "该药的审批历史"） | 问题过细导致召回不到宏观背景 |
| Query 分解 | 把复杂问题拆成多个子问题分别检索，结果汇总 | 多跳推理、对比类问题 |
| 路由（Routing） | 按问题类型分发到不同索引（FAQ / 产品文档 / 工单 / 知识图谱 / SQL） | 知识库异构，一个索引包不住 |

#### 4. 检索后处理（Post-retrieval）

- 相似度阈值过滤：低于阈值时不硬答，触发"知识库中未找到相关内容"的兜底回复，是降低幻觉最简单有效的手段；
- 上下文压缩：让 LLM 从每个块中只抽取与问题相关的句子，剔除模板、导航、版权声明等噪声；
- 去重与重排：合并相邻 / 重复块；把最相关的内容放在上下文的开头和结尾，缓解 LLM 的 "lost in the middle" 现象（对中间位置信息不敏感）；
- 元数据过滤：按部门、文档类型、时间范围、权限在**检索阶段**过滤。特别强调：权限过滤必须发生在检索期（带 ACL 条件查库），绝不能靠 Prompt 里写一句"请不要泄露无权内容"来约束。

### 代码示例：基础向量入库与检索

```python
import os
import hashlib
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import tiktoken


# ============ 1. 文档分块（Chunking）============

def chunk_document(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    将长文档切分成适合向量化的文本块。
    对于百万字级文档，建议 chunk_size=512，overlap=64。
    """
    # 使用 tiktoken 进行 token 级别的分块
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap  # 滑动窗口
    
    return chunks


# ============ 2. 加载 Embedding 模型 ============

def load_embedding_model(model_name: str = "Qwen/Qwen3-Embedding-8B"):
    """
    加载 Qwen3-Embedding-8B 模型。
    模型维度为 4096。
    首次运行会自动从 Hugging Face 下载（约16GB），请确保网络通畅和足够磁盘空间。
    """
    print(f"正在加载模型: {model_name} ...")
    model = SentenceTransformer(model_name)
    # 模型输出维度为 4096
    print(f"模型加载完成，向量维度: {model.get_sentence_embedding_dimension()}")
    return model


# ============ 3. 向量化并存入 Chroma ============

def embed_and_store(
    document_path: str,
    db_path: str = "./chroma_db",
    collection_name: str = "knowledge_base",
    chunk_size: int = 512,
    overlap: int = 64,
    batch_size: int = 100
):
    """
    读取文档 → 分块 → 向量化 → 存入 Chroma（支持断点续传和批量插入）
    """
    # 3.1 读取文档
    print(f"正在读取文档: {document_path}")
    with open(document_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"文档总字数: {len(text)} 字符")

    # 3.2 分块
    print(f"正在分块 (chunk_size={chunk_size}, overlap={overlap}) ...")
    chunks = chunk_document(text, chunk_size, overlap)
    print(f"共生成 {len(chunks)} 个文本块")

    # 3.3 加载模型
    model = load_embedding_model()

    # 3.4 初始化 Chroma 持久化客户端
    client = chromadb.PersistentClient(path=db_path)
    
    # 获取或创建 Collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
    )
    print(f"Collection '{collection_name}' 已就绪，当前文档数: {collection.count()}")

    # 3.5 批量向量化并存入
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        # 生成唯一 ID（基于文本内容的 MD5，确保可重复运行不重复插入）
        ids = [
            hashlib.md5(chunk.encode("utf-8")).hexdigest()
            for chunk in batch_chunks
        ]
        
        # 检查哪些 ID 已存在（避免重复插入）
        existing_ids = set()
        for chunk_id in ids:
            try:
                result = collection.get(ids=[chunk_id])
                if result and result.get("ids"):
                    existing_ids.add(chunk_id)
            except:
                pass
        
        # 过滤掉已存在的块
        new_chunks = []
        new_ids = []
        for chunk, cid in zip(batch_chunks, ids):
            if cid not in existing_ids:
                new_chunks.append(chunk)
                new_ids.append(cid)
        
        if not new_chunks:
            print(f"批次 {i//batch_size + 1}: 全部已存在，跳过")
            continue
        
        # 向量化
        print(f"批次 {i//batch_size + 1}: 正在向量化 {len(new_chunks)} 个文本块 ...")
        embeddings = model.encode(
            new_chunks,
            normalize_embeddings=True,  # 归一化，便于余弦相似度计算
            show_progress_bar=True
        ).tolist()
        
        # 存入 Chroma
        collection.add(
            ids=new_ids,
            documents=new_chunks,
            embeddings=embeddings
        )
        print(f"批次 {i//batch_size + 1}: 成功存入 {len(new_chunks)} 条")

    print(f"全部完成！共存入 {collection.count()} 条向量")


# ============ 4. 查询示例 ============

def query_knowledge(
    query_text: str,
    db_path: str = "./chroma_db",
    collection_name: str = "knowledge_base",
    n_results: int = 5
):
    """
    从 Chroma 中检索最相关的文档块
    """
    # 加载模型
    model = load_embedding_model()
    
    # 连接数据库
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)
    
    # 向量化查询
    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True
    ).tolist()
    
    # 检索
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    
    # 打印结果
    print(f"\n查询: {query_text}")
    print("-" * 50)
    for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
        print(f"结果 {i+1} (相似度: {1 - dist:.4f}):")
        print(f"  {doc[:200]}...\n")
    
    return results


# ============ 5. 主程序 ============

if __name__ == "__main__":
    # 配置
    DOCUMENT_PATH = "./my_long_document.txt"  # 替换为你的文档路径
    DB_PATH = "./chroma_db"
    COLLECTION_NAME = "knowledge_base"
    
    # 第一步：向量化并存储（首次运行，耗时较长）
    embed_and_store(
        document_path=DOCUMENT_PATH,
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
        chunk_size=512,      # token 数
        overlap=64,          # 重叠 token 数
        batch_size=100       # 每批处理 100 个块
    )
    
    # 第二步：查询测试
    query_knowledge(
        query_text="你的问题在这里",
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
        n_results=5
    )
```

### 代码示例：混合检索 + Rerank 精排

下面在前面入库流程的基础上，补上"BM25 稀疏召回 + 向量稠密召回 + RRF 融合 + Cross-Encoder 精排"这一生产环境最常用的组合。需要额外安装 `rank-bm25`、`jieba`：

```python
# pip install rank-bm25 jieba
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# docs / ids 为离线阶段分块后得到的文本块列表及其唯一 ID（二者一一对应）
# docs = ["...", "..."]；ids = ["md5_1", "md5_2"]
id2doc = dict(zip(ids, docs))

# 中文建议先分词再交给 BM25；英文直接 doc.split() 即可
tokenized_corpus = [list(jieba.cut(doc)) for doc in docs]
bm25 = BM25Okapi(tokenized_corpus)

# Cross-Encoder 精排模型（粗排用的双塔 embedding 模型仍沿用前文的 model）
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")


def hybrid_search(query: str, top_k: int = 30, rrf_k: int = 60):
    """BM25 与向量两路召回，RRF 融合后返回候选 ID 列表"""
    # 1) 稀疏召回：BM25
    scores = bm25.get_scores(list(jieba.cut(query)))
    bm25_ranked = [ids[i] for i in scores.argsort()[::-1][:top_k]]

    # 2) 稠密召回：向量库 ANN
    query_vec = model.encode([query], normalize_embeddings=True).tolist()
    vec_result = collection.query(query_embeddings=query_vec, n_results=top_k)
    vec_ranked = vec_result["ids"][0]

    # 3) RRF 融合：只按排名累加，不关心两路分数的量纲
    fused_scores = {}
    for rank, doc_id in enumerate(bm25_ranked, start=1):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (rrf_k + rank)
    for rank, doc_id in enumerate(vec_ranked, start=1):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (rrf_k + rank)

    return sorted(fused_scores, key=fused_scores.get, reverse=True)


def retrieve_for_generation(query: str, recall_k: int = 30, final_k: int = 5):
    """粗排融合 → Cross-Encoder 精排 → 返回最终上下文块"""
    candidates = hybrid_search(query, top_k=recall_k)

    # 4) 精排：对少量候选逐条计算 query-doc 相关性分数
    pairs = [[query, id2doc[doc_id]] for doc_id in candidates]
    rerank_scores = reranker.predict(pairs)

    top_indices = rerank_scores.argsort()[::-1][:final_k]
    return [candidates[i] for i in top_indices]


# 5) 拿到 final chunks 后，按"生成模块"一节的 Prompt 模板组装上下文并调用 LLM
context_ids = retrieve_for_generation("如何重置设备密码？")
```

## 生成模块

检索模块只决定"LLM 能看到什么材料"，生成模块决定"如何使用这些材料作答"，核心是 Prompt 组装与生成约束。一个典型的 RAG Prompt 如下：

```text
你是 XX 产品的知识库助手。请严格根据下面提供的【参考资料】回答用户问题。

要求：
1. 答案只能依据参考资料；资料中没有的内容，直接回答"根据现有资料无法确认"，不得编造；
2. 在答案中用 [1]、[2] 这样的编号标注信息来源，编号与参考资料的编号对应；
3. 如果不同资料存在冲突，优先采用发布时间更新、来源更权威的内容；
4. 先给结论，再给依据，语言简洁，不要输出与问题无关的内容。

【参考资料】
[1]（来源：product_manual.pdf 第 12 页）
……
[2]（来源：faq.md，更新时间：2026-09-01）
……

【用户问题】
{query}
```

生成阶段的关键要点：

- **角色与边界约束**：明确"只能依据上下文回答"，与检索侧的相似度阈值兜底形成双保险；
- **引用溯源（Citation）**：要求模型输出 `[n]` 标记，再映射回块的元数据（文件、页码、URL），让答案可核验，也是企业知识库的刚需；
- **上下文预算管理**：按 Rerank 分数从高到低填充上下文，总 token 不能超过模型窗口并预留生成空间，而不是把 Top-K 无脑全塞进去；
- **位置效应**：LLM 对上下文开头和结尾的信息更敏感，最相关的块要放在两边；
- **输出约束**：需要结构化输出时用 JSON Schema / function calling 约束格式；面向终端用户建议使用流式输出降低等待感；
- **拒答与降级**：检索为空、相似度低于阈值或模型自判依据不足时，统一走兜底话术，必要时转人工或触发 Web 搜索。

## RAG 的进阶范式

当朴素 / 高级 RAG 满足不了需求时，业界演化出了一批新范式：

| 范式 | 核心思想 | 适用场景 |
| --- | --- | --- |
| Self-RAG | 模型通过反思标记（reflection token）自主决定：是否需要检索、检索结果是否相关、生成内容是否被证据支持，不支持则重写 | 对事实性要求极高的问答 |
| CRAG（Corrective RAG） | 检索后用轻量评估器判断文档质量：可信则精炼使用，模糊则触发 Web 搜索补充，错误则重写查询重新检索 | 知识库覆盖不全、需要联网兜底 |
| Adaptive RAG | 路由器先判断问题复杂度：简单问题直接回答或单轮检索，复杂问题才走多轮检索策略 | 查询复杂度差异大的线上系统，兼顾成本 |
| Agentic RAG | Agent 在 ReAct 循环中自主决定"检索什么、检索几次、结果不够时怎么办"，并可调用 SQL、API、计算器等工具 | 开放域、多步骤任务、企业智能体 |
| GraphRAG | 先用 LLM 从语料中抽取实体与关系构建知识图谱，再对图谱做社区聚类和层级摘要；既能回答局部问题，也能回答全局性问题 | "这批文档的共同主题是什么"、多跳关系推理、事件脉络梳理 |
| 多模态 RAG | 图片 / 表格 / 截图用 CLIP、Cohere Embed v4 等多模态 embedding 直接向量化，或先用视觉模型生成描述（caption）再入库；表格保留结构单独解析 | 图文混排手册、财报、扫描件 |
| Long-Context 方案 | 直接把大量材料塞进长上下文窗口让模型通读 | 与 RAG 互补而非替代：少量材料通读用长上下文，海量知识精确定位用 RAG（可更新、可溯源、成本低） |

这些范式并不互斥，实际系统常常组合使用，比如"路由 + 混合检索 + Rerank + Agentic 工具调用 + 图谱辅助多跳"。

## RAG评估指标

当我们实现一个RAG系统后，还需要评估RAG系统的性能，通常会考虑以下几个指标：
- 精确率（Precision）：返回的文档中，有多少是相关的。计算的原理是：
  精确率 = 真正正例数 / 检索出来的正例数
  其中，真正正例数（TP）是返回的文档中，真正相关的文档数，检索出来的正例数（P）是返回的文档中，此次检索出来的文档数。
- 召回率（Recall）：所有相关的文档中，有多少被返回。计算的原理是：
  召回率 = 真正正例数 / 所有正例数
  其中，真正正例数（TP）是返回的文档中，真正相关的文档数，所有正例数（P）是所有文档中，真正相关的文档数。
- F1分数：F1分数是精确率和召回率的调和平均值，用于综合评估系统的性能。
  F1分数 = 2 * 精确率 * 召回率 / (精确率 + 召回率)

从RAG系统的评估指标可以看出，F1分数是检索阶段常用的综合指标，它考虑了精确率和召回率，能够综合评估检索模块的性能。但是，考虑到最终计算的结果还是取决于数据标注的精度，所以一般都需要人工参与评估。所以还需要准备一个标注数据集，用于评估RAG系统的性能。

### 检索排序指标

上面的 Precision / Recall / F1 把检索结果看成一个无序集合，而实际系统返回的是**有序的 Top-K 列表**——正确答案排在第 1 位和第 10 位，体验天差地别。因此还需要排序类指标：

| 指标 | 含义 | 典型用途 |
| --- | --- | --- |
| Hit Rate@K（命中率） | Top-K 中是否命中任一正确文档，命中记 1 否则记 0，再对所有问题取平均 | 粗排召回能力评估，最直观 |
| MRR（平均倒数排名） | 取第一个正确文档排名的倒数（1 / rank），对所有问题取平均；没命中记 0 | 只关心第一条正确结果出现在哪（问答场景常用） |
| nDCG@K | 考虑相关性等级（相关 / 部分相关 / 不相关）和位置折损的归一化增益 | 排序质量的综合评价 |
| MAP（平均精度均值） | 对每个相关文档所在位置的精度求平均，再对所有问题取平均 | 存在多个相关文档时的整体排序质量 |

### 端到端评估：RAGAS

检索指标只能衡量"找得准不准"，无法回答"最终答案好不好"。业界主流的 **RAGAS** 框架用强模型作为裁判，把 RAG 系统拆成检索和生成两个层面、四个核心维度（弱标注甚至无标准答案也能自动打分）：

| 层面 | 指标 | 衡量什么 |
| --- | --- | --- |
| 检索 | Context Precision（上下文精确率） | 召回的上下文里，相关块是否排在前面（带排序权重） |
| 检索 | Context Recall（上下文召回率） | 对照标准答案，回答所需的信息是否都被检索到了 |
| 生成 | Faithfulness / Groundedness（忠实度） | 答案中的每条陈述是否都能由上下文支撑——**衡量幻觉的核心指标** |
| 生成 | Answer Relevancy（答案相关性） | 答案是否切题，有没有跑题或堆砌无关信息 |

同类工具还有 TruLens、DeepEval、LangSmith、ARES 等。

工程上推荐的评估闭环是：

1. 人工构建一套 Golden Dataset（问题、标准答案、相关文档 ID），覆盖高频问题、长尾问题、拒答类问题；
2. 每次调整分块大小、embedding 模型、检索参数后跑一遍指标回归，用数据而不是"感觉"驱动迭代，避免"修好一个 case、退化三个 case"；
3. 上线后把命中率、拒答率、引用点击率、用户点踩率等接入监控，持续回流困难 case 补充评估集。

## 常见挑战与调优清单

| 现象 | 可能原因 | 常用手段 |
| --- | --- | --- |
| 完全检索不到答案 | 块太大语义被稀释 / 块太小语义残缺；embedding 与领域不匹配；query 与文档措辞差距大 | 调整 chunk_size 与 overlap；换领域模型；加 BM25 混合检索；上 HyDE / Multi-Query |
| 检索到了却答错 | 噪声块挤占上下文；top_k 过大；Prompt 约束弱 | 加 Rerank；做上下文压缩；减小 top_k；强化"无依据则拒答"指令 |
| 专有名词 / 型号 / 代码搜不准 | 纯向量对精确 token 不敏感 | 提高 BM25 权重或引入稀疏向量（SPLADE）；对代码做结构化切分，保留符号原文 |
| 多跳 / 全局问题答不好 | 单次 Top-K 只能拿到局部碎片 | Query 分解 + 迭代检索；引入 GraphRAG |
| 知识更新后答案仍旧 | 文档变了但索引没重建 | 按内容 hash / 更新时间做 upsert；按 source 元数据批量删除旧版本；索引版本管理 |
| 多文档信息冲突 | 不同时期、不同来源口径不一 | 元数据带时间戳和来源级别，Prompt 中声明优先级 |
| 成本高、延迟大 | top_k 大、大模型精排、重复计算 embedding | 粗排 + 精排两阶段；缓存 query 的 embedding 与检索结果；小模型做粗排；流式输出 |
| 越权与数据泄露风险 | 试图在生成阶段"叮嘱"模型别用无权内容 | 检索期就加 ACL 元数据过滤；把检索到的文档内容与用户指令明确分隔，防范间接 Prompt 注入（文档里藏"忽略以上指令"） |
| 垂直领域效果差 | 通用 embedding 不懂行业术语 | 用领域语料构造（query，正例块，难负例）三元组，对比学习微调 embedding |
| 模型编造知识库中没有的内容 | 检索为空仍强行作答 | 设置相似度阈值 + 空结果兜底话术；Prompt 强约束 + Faithfulness 指标监控 |






