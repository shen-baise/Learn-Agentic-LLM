import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

export default defineConfig({
  title: '智能体大语言模型',
  description: 'Agentic Large Language Models 中文翻译',
  lang: 'zh-CN',

  ignoreDeadLinks: true,

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '开始阅读', link: '/chapters/01_intro' },
      { text: '原论文', link: 'https://arxiv.org/abs/2610.13455' }
    ],

    sidebar: [
      {
        text: '序章',
        items: [
          { text: '第1章 引言', link: '/chapters/01_intro' }
        ]
      },
      {
        text: 'Part I 语言建模',
        items: [
          { text: '第2章 语言建模基础', link: '/part1/02_lang' },
          { text: '第3章 LLM 训练流水线', link: '/part1/03_pipeline' },
          { text: '第4章 缩放定律与效率', link: '/part1/04_scaling' },
          { text: '第5章 案例1：从零构建LLM', link: '/part1/05_case1' }
        ]
      },
      {
        text: 'Part II 推理',
        items: [
          { text: '第6章 LLM中的多步推理', link: '/part2/06_reasoning' },
          { text: '第7章 视觉、行动与世界模型', link: '/part2/07_robot' },
          { text: '第8章 元学习', link: '/part2/08_meta' }
        ]
      },
      {
        text: 'Part III 智能体',
        items: [
          { text: '第9章 智能体交互风格', link: '/part3/09_agents' },
          { text: '第10章 案例2：推理与反思', link: '/part3/10_case2' }
        ]
      },
      {
        text: 'Part IV 工具与助手',
        items: [
          { text: '第11章 智能体如何使用工具', link: '/part4/11_tools' },
          { text: '第12章 检索增强生成', link: '/part4/12_retrieval' },
          { text: '第13章 助手', link: '/part4/13_assist' },
          { text: '第14章 案例3：助手', link: '/part4/14_case3' }
        ]
      },
      {
        text: 'Part V 行为与认知',
        items: [
          { text: '第15章 行为', link: '/part5/15_behavior' },
          { text: '第16章 基于角色的交互', link: '/part5/16_role-based' },
          { text: '第17章 涌现行为', link: '/part5/17_emergent' },
          { text: '第18章 案例4：社会模拟', link: '/part5/18_case4' }
        ]
      },
      {
        text: 'Part VI 高级主题',
        items: [
          { text: '第19章 合成数据', link: '/part6/19_synthetic' },
          { text: '第20章 高级架构', link: '/part6/20_advanced_arch' },
          { text: '第21章 机械可解释性', link: '/part6/21_mechanistic' },
          { text: '第22章 认知', link: '/part6/22_cognition' }
        ]
      },
      {
        text: '结论',
        items: [
          { text: '第23章 结论', link: '/chapters/23_concl' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/' }
    ],

    footer: {
      message: '原书作者：Max van Duijn, Michiel van der Meer, Aske Plaat, Niki van Stein (莱顿大学)',
      copyright: '本翻译仅供学习交流使用 | 原文: arXiv:2610.13455'
    },

    search: {
      provider: 'local'
    },

    outline: {
      level: [2, 3],
      label: '本页目录'
    },

    docFooter: {
      prev: '上一页',
      next: '下一页'
    },

    lastUpdated: {
      text: '最后更新于'
    }
  },

  markdown: {
    config: (md) => {
      md.use(mathjax3)
    }
  },

  vite: {
    build: {
      rollupOptions: {
        onwarn(warning, warn) {
          if (warning.code === 'UNRESOLVED_IMPORT' && warning.message?.includes('/figures/')) return
          warn(warning)
        }
      }
    },
    resolve: {
      alias: {
        '/figures': '/figures'
      }
    }
  }
})
