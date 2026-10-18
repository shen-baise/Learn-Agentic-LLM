# 案例2：BALROG——LLM 推理与反思

> **学习目标**
>
> 在本案例研究中，你将获得构建面向推理领域的自反思智能体（self-reflective agent）的实践经验。具体而言，你将能够：
> - 在谜题与游戏基准测试上运行多步推理智能体（reasoning agent）。
> - 用自己的想法改进推理智能体。
> - 实现自己的自反思（self-reflective）智能体。
> - 对智能体在谜题与游戏领域中的性能与稳定性（performance and stability）进行批判性反思。

## 10.1 BALROG 概述

在本案例中，我们将使用一个能够执行自反思的智能体式大语言模型（agentic LLM）。我们使用一个流行的推理任务基准测试套件，名为 BALROG。BALROG 是 *Benchmarking Agentic LLM and VLM Reasoning On Games* 的缩写^[1]^。BALROG 基准测试由一系列难度不等的游戏组成，其中一些游戏对 LLM 来说很难玩好，正如 BALROG 论文在其基准测试研究中所展示的那样 [Paglieri et al., 2025]。最初，BALROG 游戏使用原始 LLM（vanilla LLM）进行基准测试。然而，其代码相当开放，便于编写自己的智能体方案，以检验是否能做得更好。该软件的设计允许你编写自己的上下文（强化）学习智能体代码。BALROG 的主网站是 [https://balrogai.com](https://balrogai.com)。代码易于修改和用于你自己的实验，并且设有排行榜。在本章中，我们将引导你编写自己的智能体代码，然后在 BALROG 游戏套件上运行。

> ^[1]^ Balrog 是 J.R.R. 托尔金奇幻作品中土世界的一种神话恶魔。Balrog 即使是面对最强大的巫师——灰袍甘道夫，也毫不逊色。

BALROG 的设计目的是在推理问题上挑战上下文智能体（in-context agent）。它使用谜题风格的游戏来基准测试 LLM 和 VLM 的推理能力。图 10.1 列出了这些游戏。游戏从易到难依次为：BabyAI、Crafter、TextWorld、BabaIsAI、MiniHack 和 NetHack。这些游戏提供了多种挑战：有些是基于文本的，有些包含视觉组件，所有游戏都包含对模型的推理挑战。其中一些游戏被认为是非常困难的。例如，不太可能期望你的模型、你的智能体——或者你自己——在 NetHack 上获得高分。（如果你在没有查阅攻略的情况下做到了这一点，你可能需要考虑为顶级 AI 会议撰写一篇论文了。）

![图 10.1 BALROG 中的游戏列表 [Paglieri et al., 2025]](/figures/figure_10_1.png)

### 10.1.1 谜题与游戏

BALROG 由图 10.1 所示的六个谜题和游戏组成。这些游戏从简单的网格世界谜题到困难的基于文本的龙与地下城（dungeons and dragons）策略游戏不等，具有长程信用分配（long credit assignment）片段和平坦的奖励景观（flat reward landscape）。BALROG 的作者提供了一个代码库，允许进行智能体实验、设计你自己的智能体，并观察它们在基准测试上的表现。要在 BALROG 的基准测试中取得好成绩，智能体必须擅长推理和规划，并且很可能需要善于适应和反思自身行为。

**BabyAI** 是一个简单的网格世界游戏，智能体需要完成导航任务。一个简单的强化学习（reinforcement learning）算法就能轻松解决这个游戏。任务以自然语言给出，例如：*走到蓝色球那里，然后捡起灰色钥匙*。当然，对智能体式 LLM 的挑战在于执行任务的规划部分。预计 LLM 能够在这个简单游戏上表现良好（参见 BALROG 论文）。

**NetHack** 另一方面，是一个极具挑战性的游戏，即使对人类玩家也是如此。NetHack 是一个基于文本的类龙与地下城游戏。它既包含长期战略规划——赢得一局游戏可能涉及数十万步操作，也包含短期战术战斗——抵御成群的怪物。信用分配（credit assignment）在 NetHack 中是一个挑战，因为需要理解智能体在过去采取的哪些行动对最终的成功或失败有所贡献。游戏中一个区域的行为可能在远处产生意想不到的后果，这使得强化学习智能体难以应对。众所周知，人类玩家需要数年时间才能在不借助攻略的情况下精通 NetHack。

### 10.1.2 BALROG 架构

现在我们来了解 BALROG 框架的架构。图 10.2 展示了取自论文 [Paglieri et al., 2025] 的架构图。请熟悉 BALROG 系统。阅读论文 Paglieri et al. [2025]，你可以在 [https://arxiv.org/abs/2411.13543](https://arxiv.org/abs/2411.13543) 上找到它，或访问 BALROG 网站 [https://balrogai.com](https://balrogai.com)。如果你还没有这样做，请用 `git clone` 命令在你的系统上安装 BALROG。前往 [https://balrogai.com](https://balrogai.com)，或直接前往 GitHub 仓库 [https://github.com/balrog-ai/BALROG](https://github.com/balrog-ai/BALROG)。在本地计算机上安装该仓库。决定你想使用哪个 LLM。请注意，典型的 BALROG 智能体在每一步都会进行多次 LLM 调用。如果你想在本地运行 LLM，请安装它，或从 Hugging Face 上使用它。如果你想使用（商业）云端 LLM，请获取 API 密钥并在你的环境中安装。如果有免费的学生级别（student-tier）可供申请，你可能需要考虑申请。前沿 LLM 的使用成本可能很高，但通常较旧和较小的 LLM（其中一些仍然具有相当的能力）要便宜得多。GitHub 仓库上的 BALROG readme 包含了如何安装系统以及如何为你喜欢的 LLM 安装 API 密钥的信息，参见代码清单 10.1。（在其中一个练习中，我们将使用多个 LLM 来比较它们的性能。目前，我们只是试图让系统运行起来，任何 LLM 都可以。）

```bash
export OPENAI_API_KEY=<KEY>
export ANTHROPIC_API_KEY=<KEY>
export GEMINI_API_KEY=<KEY>
```

> 代码清单 10.1：安装 API 密钥

### 10.1.3 智能体

请阅读 BALROG 中的智能体教程：[https://github.com/balrog-ai/BALROG/blob/main/docs/agents.md](https://github.com/balrog-ai/BALROG/blob/main/docs/agents.md)。在 BALROG 中，智能体是小段 Python 代码，它们通过提示词（prompt）调用 LLM 来获取如何确定下一步的答案。智能体代码负责接收模型的回答（观测，observations）并选择下一个动作（action）。LLM 用于执行各种提示词。BALROG 遵循图 9.7 中的交互风格 D，其中提示词由交互历史构建而成。

#### 10.1.3.1 提示词构建器

智能体基于先前的动作构建提示词，利用先前交互中的观测/动作对。因此，用强化学习的术语来说，智能体执行策略（policy）的功能，基于传入的观测选择下一个要执行的动作。请前往 `prompt_builder` 目录来修改提示词从动作历史中构建的方式。

#### 10.1.3.2 智能体

智能体负责记忆功能（memory function）：管理观测/动作历史。智能体必须记录过去的观测和动作，以支持上下文感知的决策（context-aware decision-making），而这必须由我们在 LLM 外部实现。BALROG 中的智能体代码查询 LLM：智能体将观测发送给 LLM，接收响应，并使用这些响应决定在游戏中采取的动作。BALROG 附带了一些预编写的智能体，你应该研究它们以设计自己的智能体。仓库中有一个朴素智能体（naive agent）和一个思维链智能体（chain of thought agent）。还提供了更健壮的版本来处理 LLM 响应。当你构建自己的智能体时，你很可能会基于自定义智能体（custom agent）进行开发。BALROG 智能体教程提供了一个简单规划智能体的示例。

## 10.2 实施步骤

我们现在将提供一些练习，帮助你探索构建自己的智能体。

### 10.2.1 复现论文

我们首先尝试复现 BALROG 论文中的结果。请查看 GitHub 上的 BALROG 代码：[https://github.com/balrog-ai/BALROG](https://github.com/balrog-ai/BALROG)。阅读评估指南（Evaluation Guide）、智能体开发教程（Agent Development tutorial），以及如何运行少样本学习（Few Shot Learning）实验（上下文学习）的说明。在你的计算机上安装相关工具和正确的 Python 版本（并将其加入 PATH）。通过在命令提示符下启动 Python 解释器来检查版本：`$ python`。首先，运行标准评估，以确保你正确安装了所有内容，参见代码清单 10.2。

```bash
python eval.py \
    agent.type=naive \
    agent.max_image_history=0 \
    agent.max_text_history=16 \
    eval.num_workers=16 \
    client.client_name=openai \
    client.model_id=gpt-4o-mini-2024-07-18
```

> 代码清单 10.2：运行标准评估

### 10.2.2 不同的 LLM

为了了解标准智能体在不同基准测试上使用不同 LLM 的表现，运行几个不同的 LLM 是有用的。建议同时运行闭源和开源 LLM，观察它们的表现。特别是来自 OpenAI、Claude、DeepSeek、OLMo 和 Qwen 的开源版本可能表现良好。运行这些模型很重要，将带来更多对各种模型相对优势的认识。

### 10.2.3 不同的谜题

接下来，为了更好地理解基准测试，在不同谜题问题上运行 LLM 智能体之后，你应该仔细查看 LLM 的提示词和回答。看看你能否发现 LLM 智能体在哪些地方失败，智能体的弱点在哪里。通过仔细检查各个动作，并更深入地理解谜题，你可能能够想出改进智能体的方法。

### 10.2.4 计算成本

BALROG 智能体在每一步都会调用 LLM，且通常使用很长的提示词。在这些谜题和游戏中，需要执行许多这样的步骤。在完整基准测试上运行不同的 LLM、使用不同的随机种子，很容易需要大量的 token。在你的实验中，首先尝试一个小而便宜的 LLM，例如 Qwen-3B。从最简单的基准测试（BabyAI 和 Crafter）开始。如果你想扩展实验，请继续，但要注意计算成本。你需要获得洞见，但也希望你的实验能在合理的时间内以合理的成本完成。

### 10.2.5 实现自己的推理智能体

现在是时候实现你自己的学习智能体了。前往 balrog 中的 `agents` 目录，编辑 `custom.py` 文件。代码清单 10.3 复现了 BALROG readme 中自定义智能体（Custom Agent）的代码。该代码可以使用代码清单 10.4 中的命令运行（指定自定义智能体，并以较少的 worker 数量温和运行）。当然，你可以在命令中指定自己喜欢的 LLM。从提供的代码开始，运行它，看看它是如何工作的。然后，你可以尝试做一些小的改进。尝试修改 LLM 的提示词，以及 Python 代码处理智能体中学习的方式——无论是在历史提示词构建器（history prompt builder）中、在计划（plan）中，还是在学习算法中。

## 10.3 评估

你的智能体的性能将由 BALROG 以进度百分比（progression percentage）的形式报告。记录每个基准测试的进度数值，以及你尝试的各种智能体类型的进度数值。以结构化的方式进行记录，你可能想编写一个 Python 或 shell 脚本来自动化实验运行和结果记录，甚至可以将进度绘制成图表。可以考虑使用 wandb.ai。哪些智能体类型表现良好？它们在哪些基准测试上表现良好？制作一个矩阵，记录你尝试的智能体类型和基准测试类型对应的进度数值。

请注意，LLM 实验本质上是非确定性的（non-deterministic）：使用相同提示词的两次运行可能会给出不同的结果。因此，对每个实验进行多次运行。如果你连续得到五个相同的结果，你可以开始信任这些结果。如果结果差异很大，你可能需要调查原因。原因是纯粹统计性的吗？降低温度（temperature）能否解决问题，还是存在其他问题？不要盲目信任报告的进度数值。由于你在计算机上使用代码，可能存在 bug 或其他问题。BALROG 系统支持深入分析运行结果。它会记录 CSV 文件，你可以用电子表格查看。查看一些运行结果，判断你所看到的是否合理，或者智能体是否表现出退化行为（degenerative behavior）。如果确实如此，尝试找出原因。是你做错了什么吗？基准测试和智能体行为之间是否存在意外的干扰？奖励景观是否太平坦，以至于智能体没有信号可以追逐，它所做的只是随机动作、毫无意义地原地打转？你能对此做些什么吗？

请注意，大语言模型领域，尤其是智能体式 LLM 领域，是一个新兴领域，我们的认识和软件框架尚未稳定。一些意想不到的结果是可以预期的，一些挑战和挫折也是如此。

```python
from balrog.agents.base import BaseAgent
import re


class CustomAgent(BaseAgent):
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.plan = None

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
            self.prompt_builder.update_observation(obs)

        plan_text = f"Current Plan:\n{self.plan}\n" if self.plan else "You have no plan yet.\n"

        planning_instructions = """
Review the current plan above if present. Decide whether to continue with it or make changes.
If you make changes, provide the updated plan. Then, provide the next action to take.
You must output an action at every step.
Format your answer in the following way:
PLAN: <your updated plan if changed, or "No changes to the plan." if the current plan is good>
ACTION: <your next action>
""".strip()

        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + plan_text + "\n" + planning_instructions

        response = self.client.generate(messages)

        # Extract the plan and action from the LLM's response
        plan, action = self._extract_plan_and_action(response.completion)

        # Update the internal plan if it has changed
        if plan != "No changes to the plan.":
            self.plan = plan

        # Save the plan in the response.reasoning field and the action in response.completion
        response = response._replace(reasoning=plan, completion=action)
        return response

    def _extract_plan_and_action(self, response_text):
        plan_match = re.search(r"PLAN:\s*(.*?)(?=\nACTION:|\Z)", response_text, re.IGNORECASE | re.DOTALL)
        action_match = re.search(r"ACTION:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)

        plan = plan_match.group(1).strip() if plan_match else "No changes to the plan."
        action = action_match.group(1).strip() if action_match else None

        return plan, action
```

> 代码清单 10.3：BALROG 中的智能体代码

```bash
python eval.py \
    agent.type=custom \
    agent.max_image_history=0 \
    agent.max_history=16 \
    eval.num_workers=64 \
    client.client_name=openai \
    client.model_id=gpt-4o-mini-2024-07-18
```

> 代码清单 10.4：运行智能体评估

![图 10.3 ReAct 示例——执行搜索 [Yao et al., 2022]](/figures/figure_10_3.png)

结构化的实验和结果记录方式是你的好帮手。规划和组织可以使实际智能体式 LLM 实验的运行、评估和改进成为一项愉快的练习。

## 10.4 自反思

当你能够复现基准测试结果，并对智能体进行了一些实验之后，你可能想要尝试一些更大的挑战。在本节中，我们将尝试将 BALROG 智能体扩展到自反思（self-reflection），其中 LLM 被用于不同的角色，如评估和动作选择。自反思，如同任何追逐自身也在被优化的目标的自适应系统一样，可能容易出现不稳定和振荡。因此，在尝试自反思想法之前，确保前一子节的基础学习系统能够正常工作并以稳定可预测的方式收敛是很重要的。请回到该部分，确保一切按预期运行。

近期有许多工作试图改进 LLM 智能体。我们将提到两个：ReAct [Yao et al., 2022] 和 Reflexion [Shinn et al., 2024]。ReAct 主要关注提示词，而 Reflexion 引入了新的智能体算法，包括短期和长期记忆。

**ReAct** ReAct 使用提示词与世界交互。ReAct 提倡使用工具来查找信息，参见图 10.3。ReAct 的原始网站是 [https://react-lm.github.io](https://react-lm.github.io)，代码仓库在：[https://github.com/ysymyth/ReAct](https://github.com/ysymyth/ReAct)。ReAct 提示方法也已在 LangChain^2^ 和 BeeAI^3^ 框架中实现。请查看这些框架，以了解 ReAct 引入的基于提示词的智能体交互风格。然后，在 BALROG 中实现基于提示词的交互。它是否提高了性能？对于哪些游戏，查找外部信息有所帮助？对于哪些游戏则不然？

**Reflexion** 如果说 ReAct 是基于提示词的、允许交互式查询的方法，那么 Reflexion 方法则引入了用 Python 编写的智能体，允许更复杂的智能体行为。Reflexion 在不同的上下文中、使用不同的提示词来使用同一个 LLM。它基于这样一个假设：批评现有提示词的问题比提出一个有效的提示词更容易。Reflexion 是一个流行的自反思架构 [Shinn et al., 2024]。Reflexion 的主页是 [https://github.com/noahshinn/reflexion](https://github.com/noahshinn/reflexion)。一篇描述其主要思想的博客文章在这里：[https://nanothoughts.substack.com/p/reflecting-on-reflexion](https://nanothoughts.substack.com/p/reflecting-on-reflexion)。图 10.4 展示了 Reflexion 的总体概念、不同的提示词以及不同类型的记忆。图 10.5 展示了 Reflexion 的代码，其中对 LLM 的不同调用在第 3、5、6 行（$M_a$、$M_e$ 和 $M_{sr}$）以及第 9、11、12 行。请注意，该代码是伪代码，关于记忆管理、提示词管理和收敛标准的重要细节尚未确定。详情请查看 GitHub 上的仓库。

![图 10.4 Reflexion 架构 [Shinn et al., 2024]](/figures/figure_10_4.png)

Reflexion 使用同一个 LLM 配合不同的提示词，这可能导致学习不稳定和循环。此外，记忆管理相对简单，存在上下文溢出（context overrun）的风险。一个想法是实现分块（chunking）或其他记忆压缩方法，可以用 Python 手工实现，也可以使用 LLM 来实现。

首先尝试使用 Reflexion 仓库，尝试让它在 Reflexion 应用（如 AlfWorld 和 HotpotQA）上运行。

![图 10.5 Reflexion 算法 [Shinn et al., 2024]](/figures/figure_10_5.png)

**扩展** 现在，尝试在 BALROG 中实现一个 Reflexion 智能体，在 BALROG 基准测试上运行。你遇到了什么问题？Reflexion 是否比标准 BALROG 智能体提高了性能？接下来，进一步扩展问题。你可以尝试在文献中找到其他反思方法，或者尝试创造自己的新反思方法。你可以尝试不同的应用类型，观察你的方法表现如何。当然，你也可以尝试其他 LLM。

> ^2^ 参见 [https://docs.langchain.com/oss/python/langchain/overview](https://docs.langchain.com/oss/python/langchain/overview)
>
> ^3^ 参见 [https://github.com/i-am-bee/beeai-framework/blob/main/python/examples/agents/react.py](https://github.com/i-am-bee/beeai-framework/blob/main/python/examples/agents/react.py)

## 10.5 总结

在本章中，我们讨论了一个实际案例研究，展示了 LLM 如何执行自反思。我们使用了 BALROG，一个流行的推理任务基准测试套件。BALROG 由从简单到困难的游戏混合组成，其中一些游戏对 LLM 来说很难玩好。BALROG 套件以开放的方式设计。我们可以编写自己的智能体代码，然后在 BALROG 游戏上运行。这是一个重要的实践案例研究。你探索了编写自己的上下文推理智能体代码。你还超越了基础学习，实现了自反思。你探索了自反思在不同 LLM 和极具挑战性的基准测试上的稳定性。
