<!--
 * @Descripttion: 
 * @version: 1.0
 * @Author: Areebol
 * @Date: 2023-06-10 23:41:59
-->
[Instruct GPT](https://www.bilibili.com/video/BV1hd4y187CR/?vd_source=ad232665e6f7da9a5121565507cf0816) 

# 自监督训练的问题
- 有效性，从文本中始终学习不到对应领域的技能
- 安全性，如何规避一些不安全的问答问题

# 方法
- 多标点数据 从无监督回到监督学习

- 希望模型能够更有帮助性
- 模型更真诚，不捏造事实
- 无害，不生成有毒数据

**使用instruction**
有标号的数据集

**使用RLHF**
基于人类反馈的强化学习
- step1: 人工标注的prompt和answer数据集，进行一个SFT的训练
- step2：输入prompt，模型输出多个可能答案，人工选择偏好，训练一个RW model
- step3：继续微调之前训练好的SFT，使得生成的答案能够得到较高分数
  
  

