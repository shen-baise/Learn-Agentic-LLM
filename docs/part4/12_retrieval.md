# 智能体 LLM 中的搜索与检索

##### 本章学习目标

在本章中，你将了解为什么检索是需要训练数据之外知识的智能体系统的一项基本能力。具体而言，你将能够：
- 描述完整的检索增强生成（Retrieval-Augmented Generation, RAG）流程，从索引和分块到查询构建和答案生成。
- 实现自适应检索方法，如 Self-RAG 和 Self-Ask，并用 Python 构建一个稠密检索系统。
- 将检索扩展到纯文本之外，涵盖图结构、代码、表格数据和多模态数据源。
- 区分智能体 RAG 与简单的单次检索，包括实时和流式变体。
- 使用标准指标评估检索系统，如 Recall@$k$、Mean Reciprocal Rank（平均倒数排名）和忠实度（faithfulness）分数。

## 12.1 检索概述

在探讨智能体 LLM 如何查阅外部知识之前，有必要将检索置于其自身更为悠久的学科历史中加以审视。信息检索（Information Retrieval, IR）是计算机与信息科学中一个成熟的领域，其历史比神经网络早数十年。许多驱动现代检索增强生成（RAG）流程的思想都是经典信息检索研究的直接延续。该领域的标准参考著作——Salton 和 McGill [1983] 以及后来的 Manning 等人 [2008]——至今仍提供着本章所依赖的概念词汇，如：索引、词项加权、排序与相关性。

### 12.1.1 词袋模型

早期的信息检索系统将文档和查询表示为词袋（bags of words），并提出了一个看似简单的问题：哪些词项使文档与查询良好匹配？Sparck Jones [1972] 用逆文档频率（Inverse Document Frequency, IDF）的概念为这个问题给出了统计答案，指出在少数文档中出现的词项比几乎处处出现的词项更具区分性。按词项在文档中出现的频率（词频，Term Frequency, TF）和其在整个语料库中的稀有程度（IDF）对词项进行加权，就产生了 TF-IDF 加权方案族，它仍然是生产搜索系统中最广泛使用的表示方法之一。词频统计词项 $t$ 在文档 $d$ 中出现的次数：

$$\text{TF}(t,d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}} \quad (12.1)$$

其中 $f_{t,d}$ 是 $t$ 在 $d$ 中的原始计数。逆文档频率遵循 Sparck Jones [1972] 的方法，对出现在许多文档中的词项进行降权：

$$\text{IDF}(t,D) = \log \frac{|D|}{|\{d \in D : t \in d\}|} \quad (12.2)$$

其中 $D$ 是文档集合。将两者结合得到文档 $d$ 中词项 $t$ 的经典 TF-IDF 权重：

$$\text{TF-IDF}(t,d,D) = \text{TF}(t,d) \times \text{IDF}(t,D) \quad (12.3)$$

### 12.1.2 向量空间模型

Salton 等人 [1975] 将这一思想形式化为向量空间模型（vector space model），将每个文档和查询表示为高维词项空间中的向量，并按文档与查询向量的余弦相似度进行排序。向量空间模型建立了一种模式，这种模式一直延续到今天的稠密检索器（见第 12.3 节）：将文档和查询归约到共享几何空间中的点，让距离或相似度代表相关性。更形式化地，在向量空间模型中，每个文档 $d$（以及查询 $q$）被表示为词项权重的向量，$\vec{d} = (w_{1,d}, w_{2,d}, \ldots, w_{n,d})$，通常使用 TF-IDF 权重（公式 12.3）。相关性通过查询向量和文档向量之间的余弦相似度来近似：

$$\text{sim}(\vec{q}, \vec{d}) = \cos(\theta) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|} \quad (12.4)$$

文档按递减的 $\text{sim}(\vec{q}, \vec{d})$ 排序。这同一操作——两个学习向量之间的余弦相似度——在稠密检索（见第 12.3 节）中再次出现，唯一的区别在于向量的获取方式。向量空间模型使用原始词项共现，而现代稠密检索器如稠密段落检索（Dense Passage Retrieval, DPR）[Karpukhin 等人, 2020] 使用学习的神经嵌入，但底层的几何直觉并未改变。

### 12.1.3 潜空间与稀疏表示

两项进一步的发展 bridging 了经典检索与现代检索。首先，Robertson 和 Zaragoza [2009] 开发了概率相关性框架（probabilistic relevance framework），它将检索视为根据词项统计估计文档与查询相关概率的问题。该框架产生了 BM25 [Robertson 等人, 1994]，一种词项加权和排序函数，对查询 $q$ 的文档 $d$ 进行如下排序：

$$\text{BM25}(q,d) = \sum_{t \in q} \text{IDF}(t,D) \cdot \frac{f_{t,d}(k_1 + 1)}{f_{t,d} + k_1 \left(1 - b + b \frac{|d|}{\text{avgdl}}\right)} \quad (12.5)$$

其中 $|d|$ 是文档 $d$ 的长度，$\text{avgdl}$ 是语料库中的平均文档长度，$k_1$ 和 $b$ 是自由参数，分别控制词频饱和度和长度归一化（典型值：$k_1 \in [1.2, 2.0]$，$b = 0.75$）。BM25 仍然是第 12.3 节混合检索讨论中引用的稀疏检索基线。

其次，Deerwester 等人 [1990] 引入了潜在语义分析（Latent Semantic Analysis, LSA），它对词项-文档矩阵应用奇异值分解以发现潜在主题，并按含义而非精确词项重叠来匹配文档。LSA 是第一个广泛使用的解决词汇不匹配问题的技术，也是今天推动稠密和语义检索的同一问题。LSA 作用于词项-文档矩阵 $X \in \mathbb{R}^{m \times n}$，其中 $m$ 是词汇表大小，$n$ 是文档数量。其元素通常为 TF-IDF 权重（公式 12.3）。应用截断奇异值分解，

$$X \approx U_k \Sigma_k V_k^\top \quad (12.6)$$

仅保留 $\Sigma_k$ 中最大的 $k$ 个奇异值，产生 $X$ 的秩-$k$ 近似。然后，文档和词项在这个降维的 $k$ 维潜空间中表示，其中语义相关的词项和文档即使没有精确的词项重叠也能彼此靠近。

从经典信息检索到本章所使用的神经检索方法的最终桥梁，是学习的稠密词向量表示的出现。Mikolov 等人 [2013] 表明，简单的神经网络架构经过训练从上下文中预测词语，可以产生捕获丰富句法和语义规律的连续词嵌入。这项工作以及随后的编码器架构，提供了稠密检索、DPR 风格的双编码器以及最终的检索增强生成所构建的表示基础模块。

换言之，检索是一个有着数十年历史的学科，关注着一个持久的核心问题：给定一个查询，大型语料库中的哪些部分与之真正相关？随着 LLM 的到来，改变的不是问题本身，而是其重要性：模型现在查阅检索到的证据，不仅仅是为人类读者排序文档，而是为了条件化自身的生成。我们现在将讨论为什么这一转变如此重要。

## 12.2 为什么检索对智能体 LLM 至关重要

在这一基础上，大语言模型引入了关注检索的新理由。LLM 是世界知识的冻结快照。LLM 的参数无法包含语料库固定之后生成的信息。当任务需要预训练时不存在的时效性、私有或长尾信息时，LLM 会表现不佳。检索增强通过在推理时查阅外部来源并基于检索到的证据条件化生成来缓解这些限制。

检索增强生成（RAG）方法已被开发出来，用于将外部知识库的数据与存储在语言模型参数中的预训练知识相整合。大多数 RAG 方法在推理时工作，尽管有些可以在微调甚至预训练期间工作。大多数 RAG 方法处理非结构化（文本）数据源。这些文本文档被索引以提高访问效率，它们可以被组织起来并创建知识图谱。最后，通常执行数据库类型的查询优化，其中查询可以被扩展或复杂查询可以被拆分为子查询 [Cong 等人, 2024]。

这个由早期系统如 REALM [Guu 等人, 2020]、DPR [Karpukhin 等人, 2020] 和 RAG [Lewis 等人, 2020] 推广的简单思想，已经发展成为一个丰富的设计空间，涵盖索引、查询规划、重排序、引用与归属以及持续数据刷新。近期的综述文章综合了这一领域的全貌和评估实践 [Gao 等人, 2023b, Huang 等人, 2024, Yu 等人, 2024, Sharma 等人, 2025, Peng 等人, 2024]。

**何时检索。** 检索在以下情况下最为有益：(1) 任务是知识密集型的（定义、实体、数字、参考文献），(2) 领域快速变化（新闻、价格、文档），(3) 需要透明性（附带引用），或 (4) 模型必须遵守严格的事实来源约束（合规门户、内部维基）。在智能体场景中，检索可以与工具使用和规划交替进行（见第 11 章）。

## 12.3 检索增强生成流程

一个实用的 RAG 系统通常实现以下七个阶段：

1. **索引（Indexing）**：索引阶段是第一步。该步骤包括：获取所有原始素材，如 HTML 页面、PDF、表格或代码，并将其转换为干净、一致的文本。保留重要结构作为元数据（标题、锚点、时间戳、作者）。将内容切分为具有有意义边界的块（如句子或段落），并保留少量重叠以免丢失上下文。创建两种可搜索的表示：稀疏（sparse）表示，如 BM25（一种经典的词项匹配方法）或 SPLADE [Formal 等人, 2021]（一种更现代的术语扩展变体），适用于精确关键词匹配；以及稠密（dense）表示，即捕获语义匹配含义的嵌入。同时存储两者使系统能在同一搜索中找到精确匹配（ID、数字）和语义匹配（释义、同义词）。

2. **查询构建（Query Construction）**：将用户的问题转化为一个或多个精确的搜索查询。这可能包括为提高清晰度而改写问题、将复杂问题拆分为子问题（多跳）、添加相关术语（查询扩展），甚至生成一个假设性答案作为搜索目标（HyDE）[Gao 等人, 2023a]。对于结构化数据源，生成特定查询如 SQL 或图查询。目标是让搜索引擎准确理解要查找什么。

3. **候选检索（Candidate Retrieval）**：使用构建好的查询从索引中拉取可能相关的块。稠密检索器（如 DPR [Karpukhin 等人, 2020]、Contriever [Izacard 等人, 2022]）按含义匹配；稀疏方法（BM25、SPLADE）按精确词项匹配；混合方法结合两者。为了在大规模下快速搜索，使用近似最近邻（Approximate Nearest Neighbor, ANN）索引如 HNSW [Malkov 和 Yashunin, 2020] 或 IVF-PQ [Jegou 等人, 2011]。对于代码或图像，使用专门的编码器。这一步的目标是在不失焦点的情况下广泛撒网。

4. **重排序（Re-ranking）**：获取检索到的候选池，使用更准确但更慢的模型（如交叉编码器，BERT、MiniLM）按相关性重新排序。ColBERTv2 [Santhanam 等人, 2022] 等晚期交互模型在质量和速度之间取得平衡。这确保排名靠前的结果最有可能回答问题。

5. **上下文组装（Context Assembly）**：去除重复项，确保覆盖查询的不同方面（多样性），并优先选择最新的或领域相关的来源。将选定的段落及其元数据（来源、URL、日期）打包成上下文包，输入 LLM。这为模型提供了一组经过筛选的、可信赖的信息集。

6. **有据生成（Grounded Generation）**：要求 LLM 使用提供的上下文生成答案，并为每个声明提供引用。如果证据不足，模型应当拒绝回答或请求澄清。某些系统如 Self-RAG [Asai 等人, 2023] 可以动态决定是否检索更多信息。重点是生成既正确又有充分支撑的答案。

7. **归属与日志（Attribution and Logging）**：在答案旁边展示来源，以便读者验证声明。记录整个过程的日志，包括查询、检索到的文档及版本，用于调试、评估和合规。这使系统透明且可审计。

一个直接的 RAG 系统遵循上述七步流程。代码清单 12.1 展示了这一过程的 Python 风格伪代码。该函数接收用户问题，将其改写为一个或多个搜索查询，检索并重排序候选段落，组装上下文窗口，最终生成答案并跟踪使用了哪些段落。我们将在下面详细介绍。

```python
# 输入：问题 q，搜索索引，段落数 k
def rag_answer(q, index, k=8):
    # 1. 构建搜索查询（可能分解多跳问题）
    queries = rewrite_question(q)
    # 2. 从索引中检索候选段落
    candidates = retrieve(index, queries, k=200)
    # 3. 使用交叉编码器或晚期交互模型重排序
    ranked = rerank(candidates)
    # 4. 从 top-k 段落中组装上下文窗口
    context = assemble(ranked[:k])
    # 5. 基于上下文生成答案并提供引用
    answer, citations = generate_answer(q, context)
    return answer, citations
```

代码清单 12.1：基本 RAG 设置的高层流程

## 12.4 自适应检索方法

传统 RAG 系统无论是否真正需要检索，都会检索固定数量的段落。这种始终开启（always-on）策略简化了工程实现，但可能损害效率和准确性。近期研究探索了自适应（adaptive）或门控（gated）检索策略，教模型决定何时搜索以及查阅多少证据。这些方法与第 6.2.4 节引入的自我反思思想一致：语言模型不是盲目地吸收检索到的文本，而是反思自身的置信度，仅在识别到知识缺口时触发检索。

具体而言，Asai 等人 [2023] 提出了 Self-RAG，它训练单个模型使用特殊的反思令牌（reflection tokens）来生成答案、请求检索和批评自身输出。当模型的草稿答案不确定时，它发出 `<RETRIEVE>` 令牌来调用搜索工具，并在下一步中整合检索到的段落。实验表明，这种学习的门控机制在避免不必要调用的同时，相比静态 RAG 提高了事实性。

一种更轻量级的方法是 Self-Ask [Press 等人, 2022]，其中模型将复杂问题分解为更简单的子问题，将它们发送给搜索工具，并使用答案组合最终响应。Self-Ask 可以与规划智能体结合，使模型能够交替进行推理、工具调用和检索。

Graph Toolformer [Zhang, 2023] 扩展了这一思想，训练一个策略将搜索查询插入程序图中。在推理时，LLM 根据当前推理状态动态决定是从文本索引、知识图谱还是数据库中检索。

除了这些具体示例外，自适应检索还可以通过以下几种方式实现：

- **基于置信度的门控（Confidence-based Gating）**：模型估计其不确定性（例如，通过熵或校准指标），当置信度低于阈值时触发检索。

- **迭代检索（Iterative Retrieval）**：系统不是一次性拉取所有证据，而是先检索少量段落，对其进行推理，然后决定是否需要额外的搜索轮次。这种方法对多跳问题很有用。

- **查询规划（Query Planning）**：智能体将用户查询分解为子任务，将它们分派给不同工具（搜索、数据库、计算器），并整合结果。检索器仅在计划需要外部知识时被调用。

- **学习检索（Learning to Retrieve）**：强化学习或监督微调可以教模型在适当时候发出 `<RETRIEVE>` 令牌。Self-RAG 是其中一个实例；其他方法探索将检索策略与生成器端到端联合训练。

自适应检索通过避免不必要的搜索来降低延迟和成本，并且通常通过聚焦于缺失知识来提高答案质量。然而，设计有效的门控策略需要仔细的评估（见第 12.7 节），并且可能需要在训练期间提供额外监督。

### 12.4.1 通过 Self-RAG 和 Self-Ask 实现自适应检索

Asai 等人 [2023] 的 Self-RAG 框架教模型在生成过程中通过发出特殊的 `<RETRIEVE>` 令牌来按需请求检索。代码清单 12.2 概述了这样一个循环如何运作。模型维护一个不断增长的上下文，并在生成输出令牌和按需拉取新证据之间交替。

```python
# 输入：问题 q，搜索索引，训练好的 Self-RAG 模型
def self_rag_answer(q, index, model):
    context = []
    output_tokens = []
    # 按序列生成令牌
    for token in model.generate_stream(q, context):
        if token == "<RETRIEVE>":
            # 模型请求更多信息
            new_docs = retrieve(index, rewrite_question(q), k=5)
            context.extend(new_docs)
        else:
            output_tokens.append(token)
        if token == "<END>":
            break
    return detokenize(output_tokens)
```

代码清单 12.2：Self-RAG 高层伪代码演示

需要注意的是，要使 Self-RAG 工作，基础模型需要经过训练或微调以产生 `<RETRIEVE>` 令牌。另一方面，Self-Ask [Press 等人, 2022] 不依赖于语言模型的特殊训练。Self-Ask 将复杂问题分解为更简单的子问题，为每个子问题检索证据，然后综合最终答案。以下高层伪代码展示了实现这一策略的一种方式：

```python
def self_ask(q, index):
    sub_questions = decompose(q)
    # 例如 "爱因斯坦什么时候出生？" 和 "诺贝尔奖是什么？"
    intermediate_answers = []
    for sq in sub_questions:
        docs = retrieve(index, rewrite_question(sq), k=5)
        answer = summarize(docs)
        intermediate_answers.append(answer)
    final_answer = compose(intermediate_answers)
    return final_answer
```

代码清单 12.3：Self-Ask 高层伪代码演示

### 12.4.2 Python 中的简单稠密检索示例

稠密检索库如 sentence-transformers 工具包使得索引小型语料库和执行语义搜索变得简单。代码清单 12.4 展示了如何嵌入句子语料库、计算与查询的相似度并显示排名靠前的结果。

```python
from sentence_transformers import SentenceTransformer, util

# 准备一个小型语料库
texts = [
    "The Eiffel Tower is in Paris.",
    "The capital of France is Paris.",
    "Berlin is the capital of Germany.",
    "The Great Wall is in China."
]

model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = model.encode(texts, convert_to_tensor=True)

def semantic_search(question, k=2):
    query_emb = model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, corpus_embeddings)[0]
    top_results = scores.topk(k)
    for score, idx in zip(top_results.values, top_results.indices):
        print(texts[idx], "(score:", score.item(), ")")

# 示例用法
semantic_search("What is the capital of France?")
```

代码清单 12.4：Python 中的简单稠密检索示例

### 12.4.3 查询改写与扩展

有效的检索通常始于将原始问题转换为索引可以匹配的形式。大语言模型可以充当查询改写器，通过释义、展开缩写、添加同义词，甚至生成假设性答案作为伪查询（HyDE 技术）。[^1] 代码清单 12.5 概述了一个简单函数，它调用 LLM 生成多种改写和扩展，然后将结果合并为一组搜索查询。

```python
def rewrite_question(q, n_variants=3):
    """
    使用 LLM 生成替代表述和假设性答案。

    :param q: 用户问题。
    :param n_variants: 要生成的改写数量。
    :return: 扩展后的查询列表。
    """
    variants = [q]
    # 要求模型生成释义和澄清
    paraphrases = llm.generate(["Rewrite: " + q] * n_variants)
    variants.extend(paraphrases)
    # 要求模型猜测答案（HyDE）并将其用作伪查询
    guess = llm.generate(["Answer briefly: " + q])[0]
    variants.append(guess)
    # 可选：从知识库中添加实体别名
    aliases = lookup_aliases(q)
    variants.extend(aliases)
    return variants
```

代码清单 12.5：查询改写的简单示例

在实践中，`rewrite_question` 函数可能调用 API（如 OpenAI 的 gpt-5）来生成释义和假设性答案，然后使用同义词库或实体链接工具添加别名。查询扩展提高了召回率，但也可能引入噪声。

## 12.5 超越文本：图结构、代码、表格数据与多模态

虽然大多数 RAG 系统处理非结构化文本，但许多智能体应用需要对结构化数据、源代码或图像进行推理。将检索扩展到纯文本之外开辟了新的机遇，但也引入了建模挑战。

**Graph-RAG** 知识图谱将实体编码为节点，将关系编码为边。在语料库上构建轻量级图结构使检索器能够跟踪链接（如"就职于"、"位于"），而不是仅依赖词袋相似度。GraphRAG 综述形式化了一个三步工作流：基于图的索引、图引导的检索和图增强的生成 [Peng 等人, 2024]。索引从文本中提取实体和关系，对其进行规范化并构建邻域结构。然后检索通过图扩展查询以找到相关节点和关联段落；这支持全局推理和消歧。最后，生成模型可以利用图结构将事实组织成连贯的叙述并避免矛盾。

### 12.5.1 基于图的检索示例

GraphRAG 系统将知识表示为实体和关系的图，并使用图遍历来检索相关节点 [Peng 等人, 2024]。一个轻量级实现可以使用 networkx 库构建。代码清单 12.6 中的示例构建了一个小型知识图谱并实现了两个简单的检索原语：(1) 获取实体的直接邻域，(2) 找到两个实体之间的最短路径。这些操作可以支持多跳推理。

```python
import networkx as nx

def build_knowledge_graph():
    G = nx.DiGraph()
    # 添加代表实体的节点
    G.add_node("Albert Einstein", type="person")
    G.add_node("Nobel Prize", type="award")
    G.add_node("Physics", type="field")
    G.add_node("1921", type="year")
    # 添加代表关系的边
    G.add_edge("Albert Einstein", "Nobel Prize", relation="received")
    G.add_edge("Nobel Prize", "Physics", relation="category")
    G.add_edge("Nobel Prize", "1921", relation="year")
    return G

def neighbours(G, entity):
    """返回实体的所有后继节点。"""
    return list(G.successors(entity))

def shortest_path(G, source, target):
    """返回最短路径上的实体序列。"""
    try:
        return nx.shortest_path(G, source, target)
    except nx.NetworkXNoPath:
        return []

# 示例用法
G = build_knowledge_graph()
print(neighbours(G, "Albert Einstein"))
# ['Nobel Prize']
print(shortest_path(G, "Albert Einstein", "Physics"))
# ['Albert Einstein', 'Nobel Prize', 'Physics']
```

代码清单 12.6：Python 中基于图的检索原语示例

在完整的 GraphRAG 系统中，图节点将存储指向原始文本段落的指针，检索过程将返回节点序列和关联段落给 LLM。图引导的检索可以通过将推理建立在显式边上并鼓励相关事实之间的一致性来减少幻觉。

**Code-RAG** 源代码具有丰富的语法结构。源代码的语义不同于自然语言。要使代码检索有效，它应在函数或模块级别进行索引，包括签名、文档字符串、抽象语法树（AST）和依赖图。CodeBERT [Feng 等人, 2020] 等模型学习代码和自然语言的联合嵌入；它们支持自然语言代码搜索、代码摘要和文档生成等任务。在 Code-RAG 中，检索器可以获取相关函数、API 和测试用例，重排序器可以使用代码感知的交叉编码器或图神经网络来优先排列精确匹配。检索到的代码片段、注释和依赖元数据随后提供给 LLM，LLM 可以回答问题、建议修复或生成新代码，同时引用源文件。评估应考虑代码的正确性、类型安全性和对依赖关系的遵守。

### 12.5.2 使用 AST 和嵌入的代码检索示例

检索源代码需要理解其语法和语义。一种简单的方法是使用 CodeBERT [Feng 等人, 2020] 等预训练模型嵌入代码片段和自然语言查询。另一种互补方法是将代码解析为抽象语法树（AST）并搜索其结构。代码清单 12.7 中的示例展示了两种方法：我们用 CodeBERT 嵌入函数以进行语义搜索，并使用 Python 的 `ast` 模块提取函数定义和文档字符串用于索引。该示例对导入模块中的函数进行索引，并执行查询与编码源代码之间的余弦相似度搜索。在实践中，可以将代码嵌入与词法特征（标识符、注释）结合，并纳入基于 AST 的匹配来处理结构化查询（如查找调用特定 API 的所有函数）。代码感知检索对于需要在工具使用过程中读取、执行或修改现有代码的智能体系统至关重要。

```python
from transformers import AutoTokenizer, AutoModel
import torch, ast, inspect

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

def encode_code(code_str):
    inputs = tokenizer(code_str, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

def extract_functions(module):
    """返回 (名称, 源码, 文档字符串) 元组列表。"""
    functions = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        source = inspect.getsource(obj)
        doc = inspect.getdoc(obj) or ""
        functions.append((name, source, doc))
    return functions

# 索引一个模块的函数
import example_module  # 你的代码库
functions = extract_functions(example_module)
embeddings = [encode_code(src) for (_, src, _) in functions]

def search_code(query, k=2):
    query_emb = encode_code(query)
    scores = [torch.nn.functional.cosine_similarity(query_emb, emb)[0]
              for emb in embeddings]
    # 返回相似度最高的 top-k 函数
    topk = sorted(zip(scores, functions), key=lambda x: x[0],
                  reverse=True)[:k]
    for score, (name, src, doc) in topk:
        print(name, "(score:", score.item(), ")\n", doc)

# 示例用法：搜索排序函数
search_code("sort a list of numbers")
```

代码清单 12.7：Python 中基于代码的检索示例

### 12.5.3 多模态与异构数据源

**表格及其他异构文档** 现有 RAG 方法在应用于包含文本和表格组件的异构文档时面临关键限制 [Yu 等人, 2025]。一个主要挑战是结构信息丢失（Structural Information Loss）。当表格数据被线性化或分块以用于基于文本的 RAG 时，其固有的关系结构——行和列之间的相互依赖性——会被破坏。这种结构丢失阻碍了 LLM 执行准确的多跳异构推理的能力。这表明仅仅将非文本数据转换为文本是不够的。

为应对这些挑战，混合框架如 TableRAG [Yu 等人, 2025] 被提出，它们统一了文本理解和表格数据的复杂操作。TableRAG 采用 SQL 编程和执行将表格相关查询视为不可分割的推理单元，从而提高了计算效率和推理保真度。类似地，MixRAG [Zhang 等人, 2025] 已被证明能显著提高异构文档 RAG 的检索性能，它要求对文本和层次化表格数据进行联合检索和推理。

这些方法成功表明，纯粹的神经（基于嵌入的）RAG 方法对于结构化数据是不够的。符号推理（如 SQL）与神经检索的整合是异构 RAG 的关键设计模式，反映了智能体 LLM 中朝向神经-符号整合以增强数据理解和保真度的更广泛趋势。

**多模态 RAG** 许多知识库包含表格、图形、图表、音频和视频。一种直接的方法是使用 OCR 提取任何文本内容，并将其余部分视为纯文本。更丰富的方法是为图像或其他模态存储嵌入（如 CLIP 或 ViT 表示），并将它们与标题和上下文配对。查询可以跨模态联合编码，允许系统检索伴随文本的相关图形或公式。例如，研究智能体可能检索一个图表及其标题来回答数据分析问题，或者教育智能体可能从教科书中获取图表。多模态 RAG 仍然是一个活跃的研究领域。挑战包括跨模态对齐和高维嵌入的高效索引。此外，非文本证据应在生成的答案中得到恰当的归属。

### 12.5.4 多模态检索示例

智能体系统有时需要检索文本之外的图像、图表或音频片段。多模态模型如 CLIP 或对比语言-音频预训练（Contrastive Language-Audio Pretraining, CLAP）将自然语言和其他模态嵌入到共享空间中。代码清单 12.8 中的高层伪代码概述了如何使用 CLIP 查找与文本查询相关的图像：

```python
from PIL import Image
import clip, torch

model, preprocess = clip.load("ViT-B/32")

def embed_image(path):
    image = preprocess(Image.open(path)).unsqueeze(0)
    with torch.no_grad():
        return model.encode_image(image)

def embed_text(text):
    text_tokens = clip.tokenize([text])
    with torch.no_grad():
        return model.encode_text(text_tokens)

# 构建图像嵌入索引
image_paths = ["cat.jpg", "dog.png", "city.png"]
image_embs = [embed_image(p) for p in image_paths]

def search_images(query, k=2):
    query_emb = embed_text(query)
    scores = [torch.nn.functional.cosine_similarity(query_emb, emb)[0]
              for emb in image_embs]
    topk = sorted(zip(scores, image_paths), key=lambda x: x[0],
                  reverse=True)[:k]
    return topk

# 示例用法：搜索城市图片
print(search_images("a skyline of a large European city"))
```

代码清单 12.8：Python 中使用 CLIP 查找相关图像的示例

多模态检索返回最相似图像的文件名。在 RAG 设置中，检索到的图像可以通过视觉编码器提取标题或物体，然后将其纳入语言模型的上下文。多模态智能体因此可以将其输出同时建立在文本和视觉证据之上。

## 12.6 智能体 RAG：超越简单检索

虽然传统 RAG 系统用外部知识增强 LLM，但它们通常是被动的，依赖于静态的、预定义的流程。一个重要的最新进展是智能体 RAG（Agentic RAG）的概念，它将自主 AI 智能体直接整合到检索增强生成流程中 [Singh 等人, 2025]。这一范式转变将 RAG 从一个单纯的工具转变为主动的、智能的、由智能体驱动的过程。智能体 RAG 系统的特征在于其能够动态管理检索策略、迭代优化上下文理解，并调整工作流以满足复杂任务需求。

智能体 RAG 代表了一种更复杂的检索循环形式，其中 LLM 智能体本身负责协调检索过程。这与更简单的 RAG 方法不同，它采用了高级智能体系统所特有的持续反馈循环。智能体不是单次的查询转换，而是可以根据中间结果和持续交互主动优化其查询。这种方法解决了传统 RAG 系统的局限性——传统系统通常受限于静态工作流，缺乏多步推理和复杂任务管理所需的适应性 [Singh 等人, 2025]。

### 12.6.1 智能体 RAG 的关键组件

智能体 RAG 框架通常包含几个使其具备动态行为的关键组件：

- **动态查询生成（Dynamic Query Generation）**：智能体基于其对任务和已检索上下文的当前理解来构建查询，从而实现更精确和有效的搜索。这可以包括响应初始检索失败的查询扩展、分解或改写。

- **迭代优化（Iterative Refinement）**：智能体评估检索到的信息的质量，如有必要，使用优化后的查询发起新的检索循环。这一迭代过程确保最终响应基于最相关和最准确的信息。

- **自主工具使用（Autonomous Tool Use）**：智能体可以选择和使用各种工具，包括不同的检索模型（如稠密与稀疏）、排序算法和外部 API，以优化其检索策略。关于智能体如何使用工具的更多信息，见第 11 章。

- **自我纠正与反思（Self-Correction and Reflection）**：智能体可以评估自身输出的准确性和完整性。如果检测到幻觉或无支撑的声明，它可以触发新的检索和推理过程来纠正错误。

智能体 RAG 的兴起表明了 RAG 在 LLM 系统中角色的转变。检索从被动的数据查找转变为主动的、智能的能力，成为智能体认知功能的组成部分，从而允许与外部知识源进行更复杂、更自适应的交互。这一转变对于开发能够处理复杂现实任务的稳健可靠的 LLM 智能体至关重要。

### 12.6.2 实时与流式 RAG

虽然传统 RAG 系统对静态数据有效，但它们通常不是为需要持续、实时信息的应用而设计的。本节探讨在持续更新的高吞吐量数据流上运行的 RAG 系统的架构考虑和挑战。此类数据流的示例包括新闻推送、社交媒体或金融市场。对于流式数据，仅仅查询更新的静态索引已不再足够。检索现在涉及动态索引管理和持续知识同化。

在针对静态数据设计的传统 RAG 系统中，预处理速度缓慢，不适合实时分析 [Zhu, 2025, Sankaradas 等人, 2025]。如何在内存预算和高数据摄取速率下维护数据新鲜度和语义覆盖？StreamingRAG [Sankaradas 等人, 2025] 等方法尝试通过实时动态演进的知识图谱来克服这些问题。

用于医学、物流和金融等动态领域的 LLM 智能体必须能够访问实时、持续更新的信息。静态 RAG 在这种快速变化的环境中会产生过时或不准确的信息。在许多智能体应用中，实时 RAG 对于高级、及时和可靠的信息是必要的。

## 12.7 评估

评估检索增强系统比评估独立语言模型更加困难，因为检索器和生成器都对最终答案有贡献。好的评估应该分离这些组件，并评判它们的联合行为。

**检索指标** 对于给定查询，我们将检索到的段落与一组标准文档进行比较。标准指标包括：

1. Recall@$k$（公式 12.7）和 Precision@$k$（公式 12.8），即 top-$k$ 中检索到的相关文档比例
2. Mean Reciprocal Rank（平均倒数排名，公式 12.9）和 Mean Average Precision（平均精度均值），捕获排序质量
3. 归一化折损累积增益（Normalized Discounted Cumulative Gain, NDCG）（公式 12.10），对更高排名赋予更大权重

BEIR 提出了一个包含 18 个数据集的异构基准测试，涵盖问答、事实核查和实体检索等任务；其实验表明，晚期交互和重排序模型提高了零样本性能，但成本也增加了 [Thakur 等人, 2021]。KILT 在统一的维基百科快照上评估知识密集型任务，同时衡量任务性能和来源追踪，鼓励模型引用其使用的段落 [Petroni 等人, 2021]。

$$\text{Recall@}k = \frac{|\{\text{相关文档}\} \cap \{\text{top-}k \text{ 检索}\}|}{|\{\text{相关文档}\}|} \quad (12.7)$$

$$\text{Precision@}k = \frac{|\{\text{相关文档}\} \cap \{\text{top-}k \text{ 检索}\}|}{k} \quad (12.8)$$

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i} \quad (12.9)$$

其中 $\text{rank}_i$ 是查询 $i$ 的第一个相关文档的位置。

$$\text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k} \quad (12.10)$$

其中

$$\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{\text{rel}_i} - 1}{\log_2(i+1)}$$

$\text{rel}_i$ 是排名 $i$ 处文档的分级相关性。

**生成指标** 上下文组装完成后，我们必须评估生成答案的质量。BLEU、ROUGE 和 BERTScore 等自动指标的用途有限，因为通常不存在唯一的标准答案。更合适的是忠实度（faithfulness）和有用性（helpfulness）。忠实度通常通过蕴含模型来衡量。对于生成输出 $y$ 和检索上下文 $x$，其工作方式如下（公式 12.11）：检查每个声明是否被引用的段落所支撑，例如通过自然语言推理模型或引文检查。

$$\text{Faithfulness} = \frac{\text{\# 被上下文蕴含的输出}}{\text{\# 总输出}} \quad (12.11)$$

有用性（或效用）表示答案是否满足了用户的信息需求。为了衡量这一点，通常需要人类评估或专门的基于 LLM 的评估器。在交互式设置中，延迟和成本也很重要。有用性基于人类或模型评分：

$$\text{Helpfulness} = \frac{\sum_{i=1}^{N} \text{score}(y_i, \text{query}_i)}{N} \quad (12.12)$$

其中 score 是 Likert 量表或成对偏好判断。

**联合评估与新指标** 一个关键挑战是分别评估检索和生成。例如，系统可能检索到不相关的文档但仍然猜对了答案，或者检索到了完美的证据但未能使用。因此，近期工作提出了统一的评估框架。Auepora 流程建议分别报告检索和生成分数，然后将它们合并为综合度量 [Yu 等人, 2024]。RAGAS 和类似工具包提供了自动化流程，使用小模型计算上下文召回率、答案正确性和引用准确性。当我们评估自适应方法（第 12.4 节）时，还需要衡量触发决策：模型在应该检索时检索的频率是多少，在不需要时跳过检索的频率又是多少？

**归属与新鲜度** 在许多应用中，模型必须引用其来源并避免过时信息。声明可以通过归属测试与证据对齐。此类测试使用自然语言推理并确保 URL 与答案的一致性。新鲜度指标衡量答案是否反映了最新的有效来源。该指标还可以量化版本漂移。归属和新鲜度对于金融或医学等时间敏感领域尤为重要。

好的系统设计基于仔细的评估。当我们发现高检索召回率但低忠实度时，说明需要更好的重排序。类似地，高忠实度但低有用性可能表明上下文覆盖不足。为了支持错误分析和审计，评估应附带查询、检索段落和模型输出的透明日志。

## 12.8 结语

检索是任何需要提供最新信息的智能体 LLM 系统的基本要素。大语言模型是知识的静态快照；RAG 帮助它们保持最新、透明和有据可依。有效的 RAG 需要在索引、查询规划、检索、重排序、上下文组装和有据生成等环节进行精心的工程实践，同时需要自适应技术让模型决定何时以及如何搜索。我们还看到了检索如何扩展到纯文本之外，涵盖图结构、代码和多模态资源。对于构建可信赖的系统，严格的评估至关重要。

**运维考虑** 在生产系统中，检索器是一个活跃的服务，而非一次性脚本，需要仔细维护。爬虫、Webhook 和文档管道应持续摄取新数据。注意跟踪版本和废弃情况，维护每个实体的时间线以优先使用最新事实。系统应在检索时应用访问控制列表，并在索引阶段编辑个人身份信息。延迟和成本很重要：缓存嵌入和 top-$k$ 结果，在对话轮次间复用检索，通过抽取式摘要压缩上下文，按领域调整 $k$ 和块大小，仅对排名靠前的候选进行重排序。

**实践建议** 创建 RAG 系统时，从混合方法开始：将强力的稠密检索器与 BM25 或 SPLADE 结合，使用交叉编码器对前一百个命中进行重排序，然后将六到十二个段落发送给 LLM。仔细地分块语料库，例如使用 300-500 个 token 的窗口和 10-15% 的重叠，保留标题和表格，同时附加时间戳或其他元数据。对于查询编写，精心设计查询。改写复杂问题，分解多跳查询，扩展实体别名并强制执行领域过滤。为了透明性，始终要求引用。然而，如果证据薄弱，则提出后续问题或拒绝回答。最后，使用基于证据的系统设计，衡量你所构建的。你应该跟踪检索召回率、有据性、时效性和成本。另一个建议是尝试重排序器。为了调试，你应该记录系统追踪。

##### 动手实践！

想要尝试检索增强生成？打开我们的：RAG 教程（Google Colab）

**融入智能体 AI** 检索是智能体 LLM 的组成部分，用于规划、自我反思以及在现实世界中行动时（见第 13 章）。智能体管道可以根据模型的置信度或计划动态调用检索器，如 Self-RAG 和 Graph Toolformer 所示。它们可以查询不同的数据源，如文本、图结构、代码和图像，并利用每种数据源的优势。模型与其记忆之间的界限因此变得模糊：RAG 成为智能体认知循环的一部分。

**延伸阅读** 如需深入探索，请参阅关于 RAG、评估和基于图方法的综述文章 [Gao 等人, 2023b, Huang 等人, 2024, Yu 等人, 2024, Sharma 等人, 2025, Peng 等人, 2024]。基础性工作包括 REALM、DPR 和原始 RAG 公式 [Guu 等人, 2020, Karpukhin 等人, 2020, Lewis 等人, 2020]。GraphRAG、评估和智能体 RAG 社区正在快速发展；关注其最新论文和基准测试是保持前沿的最佳方式。

[^1]: HyDE 代表假设文档嵌入（Hypothetical Document Embeddings），它生成一个合理的答案，然后将其嵌入并用于检索支撑段落 [Gao 等人, 2023a]。
