# 智能体大语言模型 — Agentic Large Language Models 中文翻译

> 莱顿大学研究生课程教材 | 在线阅读：https://shen-baise.github.io/Learn-Agentic-LLM/

## 关于本书

本书由莱顿大学四位教授基于其讲授的研究生课程整理而成，全面介绍了**智能体大语言模型（Agentic LLMs）**的理论基础、技术架构和应用实践。

| 信息 | 详情 |
|------|------|
| **作者** | Max van Duijn, Michiel van der Meer, Aske Plaat, Niki van Stein |
| **机构** | 莱顿大学 (Leiden University) |
| **原文** | [arXiv:2610.13455](https://arxiv.org/abs/2610.13455) |
| **规模** | 472 页 · 7 大部分 · 23 章 · 137 张插图 · 4 个动手案例 |

## 全书结构

| 部分 | 主题 | 章节 |
|------|------|------|
| **序章** | 引言 | Ch 1 |
| **Part I** | 语言建模 | Ch 2-5（含案例1：从零构建LLM） |
| **Part II** | 推理 | Ch 6-8（多步推理、视觉与世界模型、元学习） |
| **Part III** | 智能体 | Ch 9-10（含案例2：BALROG推理与反思） |
| **Part IV** | 工具与助手 | Ch 11-14（含案例3：多智能体诊断） |
| **Part V** | 行为与认知 | Ch 15-18（含案例4：社会模拟） |
| **Part VI** | 高级主题 | Ch 19-22（合成数据、高级架构、可解释性、认知） |
| **Part VII** | 结论 | Ch 23 |

## 课程适用场景

本书提供了三种课程配置方案：

- **综合课程**：涵盖全书 Parts I-V，适合完整的研究生学期课程
- **智能体助手方向**：聚焦 Parts II-IV，适合侧重应用开发的课程
- **社会模拟方向**：聚焦 Parts II-III + V，适合侧重行为研究的课程

## 先决条件

读者需要具备本科水平的以下知识：
- Python 与 PyTorch 编程
- 机器学习基础（分类/回归、过拟合/欠拟合）
- 深度学习基础（神经网络、反向传播、CNN、RNN）
- 强化学习基础（Q-learning、DQN、PPO）
- 自然语言处理基础（分词、词嵌入、注意力机制）

## 本地运行

```bash
npm install
npm run dev       # 本地开发预览
npm run build     # 构建静态站点
npm run preview   # 预览构建结果
```

## 关于本翻译

本翻译基于原论文 arXiv:2610.13455v1，仅供学习交流使用。翻译采用 AI 辅助逐章精译，关键术语首次出现时保留英文原文注释。

## License

原论文为开放获取。本翻译仅供学术教育和学习参考。
