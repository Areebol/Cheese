
# [GPT系列论文](https://www.bilibili.com/video/BV1AF411b7xQ/?vd_source=ad232665e6f7da9a5121565507cf0816)

## GPT1 - Improving Language Understanding by Generative Pre-Training

### abstraction
**痛点**
NLP没有大规模的label数据集

**method**
利用无监督文本训练

### introduction
**难点**
- 如何确定目标函数
- 如何迁移到下游任务

**方法**
半监督学习->有监督微调
这类方法最后归类为自监督学习（半监督学习）

**特点**
transformer能够在自监督学习中学习到更稳健的feature
天然有迁移能力

### framework

#### 如何做预训练
根据上下文窗口大小+模型->最大似然对下个token的预测概率

**相比BERT**，GPT的技术路线更难，单方向的预测

#### supervised fine-tuning
给定序列，预测label

将预训练的目标函数作为附加目标，效果会更好

#### task-specification 
在多个任务对模型做一丢丢的改动，再用特定数据集来fine-tune

### 模型架构
12层decoder
12头注意力
768维的token

---

## GPT2 - Language Models are Unsupervised Multitask Learners

### abstraction
- 创建新的数据集 WebText
- 1.5B的模型
- ZeroShot实验

### Introduction
- 主流：在一个任务上收集特定数据集
- 原因：模型的泛化能力还不够好
- 方法：还是使用GPT，训练一次，泛化到更多的任务，使用ZeroShot方法


### Method
引入了prompt
对下游任务的泛化在开头使用prompt确定任务类型

---

## GPT3 - Language Models are Few-Shot Learners

### abstraction
直接训练了一个17.5B的模型

引入了few-shot的概念，认为模型在训练的时候能够根据context学到具体任务的特征

加入了对模型的社会影响问题，模型的生成内容可能会污染社会，带来安全问题