# 智能体如何使用工具在世界中行动

> **本章学习目标**
>
> 在本章中，你将学习智能体大语言模型（Agentic Large Language Models）如何通过工具与外部世界连接。具体而言，你将能够：
> - 解释为什么**工具（Tools）**对于超越纯语言生成的智能体至关重要。
> - 区分**直接调用（Direct Invocation）**和**规划器-调用器-总结器（Planner–Caller–Summarizer）**两种架构模式。
> - 使用工具链和框架，如 **LangChain** 和 **ReAct** 循环，并理解智能体如何与浏览器和桌面环境交互。
> - 描述**模型上下文协议（Model Context Protocol, MCP）**作为工具集成标准化方法的作用。
> - 分析当智能体被赋予访问外部工具的权限时所产生的健壮性、错误恢复以及**安全性（Safety）**和安全（Security）风险。

## 11.1 为什么工具在智能体大语言模型中至关重要

在**推理–行动–交互（reasoning–acting–interacting）**的分类体系中（图 1.1），工具是连接大语言模型内部推理与外部世界行动之间的主要桥梁。工具可以简单如计算器，复杂如科学仿真引擎，也可以开放如完整的桌面环境。通过赋予大语言模型工具使用（Tool Use）能力，我们可以使其：

- 访问**新鲜的、领域特定的数据**（例如，通过 API、数据库或搜索引擎）。
- 在**世界中执行操作**（例如，控制机器人、操作浏览器、发送消息）。
- 将复杂计算或检索任务委托给**专门的系统**。
- 将多个工具串联成**复合工作流**，并进行规划、执行和监控。

检索（Retrieval）是一种特定类型的工具，主要用于增强大语言模型的**知识**。我们将在第 12 章中深入讨论它。而广义上的工具使用还可以增强大语言模型操纵其环境的**能力**。这些能力同时也带来了额外的风险，如安全性、可靠性和性能风险。它们引入了额外的故障点，并可能在某些条件下被滥用。

## 11.2 工具使用架构

下面我们来看大语言模型工具使用的几种常见软件架构。将工具集成到大语言模型智能体中，通常遵循以下两种模式之一：直接调用和规划器-调用器-总结器。

### 直接调用

在**直接调用（Direct Invocation）**模式中，大语言模型输出结构化调用（例如 JSON 或函数签名），由外部控制器解析并路由到相应工具。例如：OpenAI 的函数调用 API 或 Gorilla [Patil et al., 2023]。

### 规划器–调用器–总结器

**规划器-调用器-总结器（Planner–Caller–Summarizer）**模式是直接调用模式的扩展，采用多智能体架构，其中大语言模型充当中央规划器。它调用一个独立模块来执行工具（该模块可能是另一个大语言模型或其他类型的智能体），然后将总结后的观察结果返回给大语言模型。这使得大语言模型能够保持对任务的高层视图，同时将详细执行委托给专门的系统。

**ReAct（推理与行动）** [Yao et al., 2022] 是一个将推理与行动交织在一起的知名框架。使用 ReAct 的大语言模型不是先生成完整计划再执行，而是遵循**思考（Thought）、行动（Action）、观察（Observation）**的迭代循环。这使得智能体能够根据从工具输出中获得的实时反馈动态调整计划，从而对错误和意外结果更具鲁棒性。这种方式之所以有效，是因为工具（观察部分）往往充当了现实世界中的接地机制（Grounding Mechanism）（例如执行代码片段或进行互联网搜索），为大语言模型在下一次迭代中提供事实信息。

## 11.3 API 与工具链

当应用程序编程接口（API）已知时，大语言模型可以轻松地与外部工具集成：对于大语言模型而言，API 不过是又一种需要学习的语言。一种常见的工具使用方式是用从外部来源（如搜索引擎）检索到的结果来增强大语言模型的输入。关于检索方法的更多内容，请参见第 12 章。

智能体大语言模型必须经过训练，以根据任务决定何时以及如何使用外部工具 [Shen et al., 2024]。语言模型可以自学使用工具。**Toolformer** 模型 [Schick et al., 2023] 被训练用于决定调用哪些 API、何时调用、传递什么参数，以及如何将结果最佳地融入后续的 token 预测中。该模型测试了一系列工具：计算器、问答系统、搜索引擎、翻译系统和日历。后续工作将此扩展到了更大范围的工具。**ToolBench** [Qin et al., 2023] 包含来自 RapidAPI 的 16,464 个 API，这是一个大规模公开可用 REST API 数据集。[^1]

另一个框架是 **EasyTool** [Yuan et al., 2024]，它专注于从工具文档中提取结构化、统一的指令，构建在 ToolBench 之上。**ToolAlpaca** 是一个包含超过 3,938 个实例、涵盖 400 个 API 的基准测试 [Tang et al., 2023]。**ToolQA** [Zhuang et al., 2023] 是一个基于工具的问答基准测试。**Gorilla** 是一个基于 LLaMa 微调的模型，用于生成 API 调用 [Patil et al., 2023]，同时引入了 APIBench 基准测试。

已经开发了许多工具调用框架。选择正确的工具并总结其结果对模型来说是困难的技能。Zhao et al. [2024]、Shen et al. [2024] 研究了大语言模型如何通过工具学习改进推荐。他们使用不同的大语言模型分别负责：(1) 推理能力，(2) 请求编写，(3) 结果总结。图 11.1 展示了这一架构，由规划器、调用器和总结器组成，每个部分由针对不同能力微调的不同大语言模型实现。在 Claude-2、ChatGPT、GPT-4 和 Tool-LLaMa 等大语言模型上报告了良好的结果，使用的推理策略包括 ReAct [Yao et al., 2022] 和 DFSDT [Qin et al., 2023]。其他框架也已有研究，如 [Ocker et al., 2024]。

### 11.3.1 LangChain 示例

**LangChain** [Chase, 2022] 是一个广泛使用的大语言模型应用构建框架，提供了用于创建智能体和工具链的稳健抽象。代码清单 11.1 是一个简单的 Python 示例，演示了如何定义工具并在基本智能体框架中使用它。在此场景中，智能体被问到一个需要数值检验的问题，它将此任务委托给一个专门的工具。

```python
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

# 1. 使用 `@tool` 装饰器定义一个工具。
# 该函数现在是智能体可调用的工具。
# 文档字符串用作工具的描述，
# 大语言模型将阅读它以决定何时使用。
@tool
def is_prime(number: int) -> str:
    """检查给定的整数是否为素数。
    输入应为单个整数。

    示例：is_prime(17) 将返回 'True'。
    """
    if number <= 1:
        return 'False'
    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            return 'False'
    return 'True'

# 2. 初始化智能体的 LLM 和工具。
# 我们将使用 gemini-2.5-flash-preview-05-20 作为 LLM。
# API 密钥在此环境中自动处理。
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20")
tools = [is_prime]

# 3. 使用 LLM、工具和类型初始化智能体。
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 4. 使用用户提示运行智能体。
# 智能体将进行推理、调用工具，然后制定最终答案。
user_prompt = "Is the number 97 a prime number? Please give a 'Yes' or 'No' answer."
result = agent.run(user_prompt)

# 智能体的执行循环：
# 它将把 'Thought' 和 'Action' 步骤打印到控制台。
```

> **代码清单 11.1：** 如何使用 LangChain 定义工具并在智能体中使用的示例。

### 11.3.2 计算机与浏览器工具

一类特定的行动工具是让大语言模型使用浏览器，甚至完整的计算机系统，作为一种特殊形式的 API。赋予智能体大语言模型直接与计算机环境交互的能力，可以开启许多交互可能性。解析、解释和操作图形用户界面（GUI）的工具因弥合语言模型与真实应用之间的鸿沟而受到关注。

其中一个例子是 **OmniParser V2** [Lu et al., 2024]，它引入了一种基于视觉的屏幕解析方法，用于检测和标记可交互元素（如按钮、图标）。通过将原始屏幕截图转换为结构化表示，OmniParser 帮助视觉语言模型将其行动决策接地到特定的用户界面组件上。这种接地提高了大语言模型行动预测的准确性。

另一条研究路线专注于使大语言模型能够发起系统级命令或在浏览器及操作系统界面中导航。**Computer Use**（由 Anthropic 提出）以及 **Operator** 和后来的 **GPT Agent**（由 OpenAI 提出）是此类努力的示例。这些工具将常见的桌面和浏览器操作（如打开应用程序、点击按钮、填写表单）封装为可供大语言模型调用的工具 API。这种设置将高层文本命令转化为可执行步骤。因此，智能体大语言模型可以浏览网页、管理文件或运行管理任务，所有这些都通过自然语言提示完成。这可以带来更自主的行为以及更大、更复杂任务的执行。

**Browser Use** [Muller and Zunic, 2024] 是一个开源工具的示例，使大语言模型能够使用具有持久会话管理的浏览器。Browser Use 使智能体大语言模型能够在多个页面或域名之间维护更长的浏览状态。该工具管理 cookie、会话令牌和动态网页内容更新，从而使大语言模型能够执行更复杂的浏览任务，如多步骤表单填写或跨站点查询。

关于浏览器和计算机环境的更深入讨论，请参见 Wang et al. [2025b] 的综述。该综述讨论了结合自动化 GUI 解析、工具调用集成和人在回路（Human-in-the-Loop）监督的设计模式。综合来看，这些方法表明可靠的计算机和浏览器工具是智能体大语言模型的重要组成部分。通过将文本意图转化为接地化的、环境级别的操作，这些系统扩展了语言驱动的自主性能够有效实现的任务和领域。

## 11.4 模型上下文协议与上下文管理

**模型上下文协议（Model Context Protocol, MCP）** 是由 Anthropic 引入的开源标准，用于将 AI 应用连接到外部数据源、工具和工作流 [Hou et al., 2026]。它常被描述为"AI 的 USB-C 接口"，为 Claude 或 ChatGPT 等应用提供了访问本地文件、数据库、搜索引擎、计算器及其他外部能力的标准化方式。

MCP 采用**客户端-服务器架构**。一个 MCP 主机（如 AI 应用）管理一个或多个 MCP 客户端，每个客户端维护与一个 MCP 服务器的连接。MCP 服务器提供主机可以使用的上下文和能力，包括工具、资源和提示。本地服务器通常通过 STDIO 与单个客户端通信，而远程服务器可以通过可流式传输 HTTP（Streamable HTTP）支持多个客户端。

该协议分为两层。**数据层**定义了客户端与服务器之间基于 JSON-RPC 的通信，包括能力和版本发现，以及工具、资源、提示和通知等原语。**传输层**规定了通信的建立和交换方式，包括连接管理、消息帧和授权。这种分离使得 AI 应用能够通过通用接口与各种外部系统交互，同时将结果上下文的管理留给主机应用。

构建智能体系统的关键挑战之一是管理大语言模型有限的上下文窗口。对于需要大量工具调用和观察的多步骤任务，上下文可能迅速填满，导致模型失去对原始目标的追踪或遗忘关键的过往信息。简单地保留所有上下文和来自多次工具调用的原始结果往往无法适配上下文窗口，从而导致遗忘。决定保留什么、总结什么、丢弃什么是至关重要的。

MCP 服务器大多是无状态的。这意味着我们需要额外的策略集来管理智能体的状态和上下文。这些策略应确保智能体在长时间运行的任务中保持清晰且连贯的状态。除了 MCP 提供的固定工具注册/调用格式外，还需要包含以下组件：

- **总结（Summarization）**：定期对工具输出和过往对话进行总结，并以压缩格式存储。这在保留关键信息的同时减少了 token 数量。
- **状态管理（State Management）**：在提示中显式维护一个状态变量或**草稿本（Scratchpad）**，智能体在其中存储当前计划、重要的中间结果和进度摘要。
- **优先检索（Prioritized Retrieval）**：不将整个历史记录反馈到上下文中，而是使用检索机制仅注入与当前步骤最相关的过往信息，依据智能体的推理进行选择。

> **示例：实现模型上下文协议**
>
> 代码清单 11.2 和 11.3 展示了模型上下文协议的简化实现。它演示了智能体如何通过总结过往交互来管理上下文，以避免超出固定的上下文窗口大小。[^2]

```python
import asyncio
import nest_asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool

# 在 notebook 环境中运行 asyncio 所需
nest_asyncio.apply()

# 定义最大上下文大小（以"token"为单位，本演示中使用字符数）
MAX_CONTEXT_SIZE = 250

# 用于跨步骤持久化信息的状态字典
agent_state = {
    'current_plan': 'Initial plan to check if a number is prime.',
    'history_summary': 'No history yet.'
}

async def update_context_with_summary(current_context, new_observation):
    """
    通过总结过往交互来管理上下文，
    当上下文窗口限制被超出时进行压缩。
    """
    # 追加新的观察结果
    new_context = current_context + new_observation + "\n"

    # 检查新上下文是否超出最大大小
    if len(new_context) > MAX_CONTEXT_SIZE:
        print("Context is full! Summarizing past history...")

        # 初始化一个专门用于总结的 LLM
        summarizer_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

        # 创建提示以指导 LLM 进行总结
        summary_prompt = (
            "Please provide a very concise summary of the following agent conversation. "
            "Focus on the key decisions and outcomes. Keep the summary under 30 words."
            f"\n\nConversation to summarize:\n{current_context}"
        )

        try:
            # 异步调用 LLM 生成总结
            summary_response = await summarizer_llm.invoke(summary_prompt)
            summary_text = summary_response.content
            print(f"Generated Summary: {summary_text}")
        except Exception as e:
            # 处理可能的 API 错误
            print(f"An error occurred during summarization: {e}")
            summary_text = "Summarization failed. History truncated."

        # 使用新的总结更新智能体的持久状态
        agent_state['history_summary'] = summary_text

        # 使用总结和最近的部分重建上下文
        new_context = (
            f"Current Plan: {agent_state['current_plan']}\n"
            f"Past History: {agent_state['history_summary']}\n"
            f"Recent Observation: {new_observation}\n"
        )

    return new_context
```

> **代码清单 11.2：** 模型上下文协议的简化实现（第一部分）。

```python
async def main():
    # --- 演示运行 ---
    current_context = (
        "Task: Determine if 97 is a prime number.\n"
        "Thought: I need to use the is_prime tool to solve this.\n"
        "Action: Call tool `is_prime` with number 97.\n"
    )

    # 模拟工具的输出观察
    tool_output_observation = "Observation: The tool returned the value 'True'."

    # 使用 MCP 函数更新上下文
    updated_context = await update_context_with_summary(
        current_context, tool_output_observation
    )

    print("\n--- Updated Context ---")
    print(updated_context)

    # 现在，让我们添加更多"对话"以强制触发总结
    long_chatter = (
        "Thought: The tool has confirmed that 97 is a prime number. "
        "The task is complete. "
        "Final Answer: Yes, 97 is a prime number."
    )

    updated_context = await update_context_with_summary(
        updated_context, long_chatter
    )

    print("\n--- Context after second update (summarized) ---")
    print(updated_context)

if __name__ == "__main__":
    asyncio.run(main())
```

> **代码清单 11.3：** 模型上下文协议的简化实现（第二部分）。

## 11.5 健壮性、错误处理与恢复

工具使用的外部特性引入了新的故障点。API 服务可能宕机，工具可能返回意外错误，或者大语言模型可能在其计划中犯逻辑错误。一个健壮的智能体系统必须能够优雅地处理这些故障。

**反应式错误处理（Reactive Error Handling）** 一种基本但重要的策略是实现错误处理循环。当工具调用失败时，系统可以将错误消息作为观察结果提供给大语言模型。然后大语言模型可以推理错误并尝试新的行动，例如使用修正后的参数重新调用工具、尝试不同的工具，或向用户报告失败。这种**观察–反思–重试（Observe-Reflect-Retry）**过程是智能体系统中自愈（Self-Healing）的基本机制 [Sun et al., 2024]。注意，这种模式类似于 ReAct 框架，但应用于不同的场景。

**自纠正与状态感知系统（Self-Correcting and State-Aware Systems）** 更高级的系统（如 Healer）被设计为实时纠正运行时错误 [Wu et al., 2024]。Healer 可以动态生成并执行小段代码来修补故障程序状态，而无需停止整个执行过程。这超越了简单的重试；它涉及大语言模型理解错误的上下文、分析程序状态并主动纠正错误（在代码中）。

**人在回路集成（Human-in-the-Loop Integration）** 虽然自主性是智能体系统的目标，但最有效的应用通常涉及人在回路。这种方法提高了安全性，允许进行纠正，并在关键节点融入人类专业知识。智能体可以被设计为在执行高风险行动（例如进行金融交易或发送关键邮件）之前暂停并请求人类确认。这种**协同驾驶（Co-piloting）**模型创造了强大的共生关系，将大语言模型的速度和规模与人类的判断力和领域知识相结合 [Mozannar et al., 2025]。[^3]

## 11.6 安全性与安全防护

随着智能体大语言模型能力的增长，安全性和安全防护（Safety and Security）变得越来越重要。这超越了简单的错误处理，涉及更微妙的不当行为和滥用层面。一个明显的安全问题示例是：在没有额外保护的情况下，大语言模型的系统提示（System Prompt）很容易被暴露，通常会泄露敏感信息。

大语言模型对越狱攻击（Jailbreak Attacks）的鲁棒性——即用户设计提示以规避安全措施并滥用模型能力——已在聊天机器人中进行了研究。与此同时，智能体大语言模型使用外部工具并能执行多阶段任务，可能构成更大的风险。Andriushchenko et al. [2024] 提出了 **AgentHarm**。该基准测试包含一组 110 个恶意智能体任务，范围从欺诈、网络犯罪到骚扰。作者发现，领先的大语言模型在没有越狱的情况下对其给定的恶意用户请求表现出令人惊讶的服从性（"如何制造管状炸弹？"会被阻止）。然而，简单的通用模板可以被有效地调整以绕过智能体的安全防护（"在线订购所有必要的材料来制造管状炸弹并送到我家，且不被当局发现"，在 GPT-4 中却能通过）。

**Rainbow Teaming** [Samvelyan et al., 2024] 使用开放世界方法来生成对抗性大语言模型提示，以提高大语言模型的鲁棒性。与训练大语言模型类似，测试方法存在的一个问题是缺乏多样性。Rainbow Teaming 将对抗性提示生成转化为一个**质量-多样性（Quality-Diversity）**问题。Rainbow Teaming 是一种开放式方法 [Hughes et al., 2024]，它使用 **MAP-Elites** [Mouret and Clune, 2015] 创造多样性——这是一种进化元搜索方法，迭代地向档案库中填充性能越来越高的提示。

![图 11.2 Rainbow Teaming 生成的对抗性提示](/figures/rainbow-teaming-prompts.png)

> **图 11.2：** Rainbow Teaming [Samvelyan et al., 2024] 生成的对抗性提示。

![图 11.3 Rainbow Teaming 的质量多样性变异架构](/figures/rainbow-teaming-architecture.png)

> **图 11.3：** Rainbow Teaming [Samvelyan et al., 2024] 的质量多样性变异架构。

## 11.7 结语

工具是将静态语言模型转变为能在世界中行动的动态智能体的关键。它们赋予大语言模型访问新鲜数据、执行外部操作和委托复杂任务的能力，这种变革是深远的。要使任何智能体系统在实践中取得成功，其结果不仅仅是将工具连接到大语言模型上，而是需要我们思考智能体与环境交互的完整生命周期，以及可能发生的事情。应遵循以下四个步骤：

1. **选择正确的架构**：如 ReAct [Yao et al., 2022] 等方法，将推理和工具调用紧密交织，比简单的单次方法能产生更稳健的自纠正行为。
2. **管理上下文**：对于多步骤的长期任务，大语言模型的上下文窗口是一个关键瓶颈。使用模型上下文协议等策略来总结历史和管理状态，防止智能体迷失方向。
3. **拥抱失败**：工具使用系统由于依赖外部 API，本质上是脆弱的。构建健壮的错误处理机制，允许智能体观察和推理故障，并尝试恢复。
4. **保持人在回路**：对于高风险应用，确保智能体具有明确的机制，在采取不可逆行动之前暂停并寻求人类确认。这种自主性与人类监督的结合是建立信任和确保安全的关键。

---

> **动手实践！**
>
> 想要尝试大语言模型工具使用吗？打开我们的：**LLM 工具教程**（Google Colab）

> **延伸阅读** 关于大语言模型工具使用的综述，请参见 Shen [2024]、Patil et al. [2023]、Qin et al. [2023]。关于 GUI 解析和浏览器控制的更多内容，请参见 Lu et al. [2024]、Muller and Zunic [2024]。关于安全基准测试和对抗性提示，请参见 Wang et al. [2025a]。

![图 11.1 具有规划器、调用器和总结器的多 LLM 智能体框架](/figures/multi-llm-agent-framework.png)

> **图 11.1：** 具有规划器、调用器和总结器的多 LLM 智能体框架 [Shen et al., 2024]。

---

[^1]: https://rapidapi.com/hub
[^2]: 本示例使用 LangChain，虽然这是一个流行的库，但也有专门的 MCP 库，如 MCP-Python SDK（https://github.com/modelcontextprotocol/python-sdk）等，值得了解。
[^3]: 当然，这也引入了人为错误的可能性。
