<!--
 * @Descripttion: 
 * @version: 1.0
 * @Author: Areebol
 * @Date: 2023-06-10 23:41:47
-->
[OpenAI-training](https://www.bilibili.com/video/BV1ts4y1T7UH/?vd_source=ad232665e6f7da9a5121565507cf0816)

# GPT assitant training pipeline
## Pretraining
99% training time
- 收集混合数据集合（无标记文本）
- tokenization，将raw text转化为integers
  - Raw Text
  - Tokens
  - Integers

模型超参数
- layer层数
- attention头数
- dimension维度

训练成本
- meta的LlaMa大概 $2048A100*21days$

训练过程
- input：(B,T)指代batch size和maximum token length

完成这个阶段得到base model，此时这个model完全是一个document filler
## Supervised Finetuning
收集一个小的SFT数据集

完成这个阶段得到sft model，此时这个model已经可以作为一个AI assitant

## Reward Modeling
RM dataset

用SFTmodel对prompt生成多个回答，人工对回答rank

rank完的数据加上sft数据得到一个RM dataset，然后使用这个dataset训练一个RW model
## Reinforcement Learning
使用RW model对SFT model进行RL

最后得到的模型就是RLHF model

## models 比较
base model的generate content熵更高，更发散
RLHFmodel的gc就更focus

## Chain of Thought
人脑对句子的生成，经过思维链逐步推导
LLM对句子的生成，只是通过对训练文本的不断采样，尽力去预测下一个token

对system2的recreate
