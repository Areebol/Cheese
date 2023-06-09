<!--
 * @Descripttion: 
 * @version: 1.0
 * @Author: Areebol
 * @Date: 2023-06-09 13:13:30
-->
[Hung-yi Lee] ChatGPT 原理剖析 [Youtube](https://www.youtube.com/watch?v=yiY4nPOzJEg&list=RDCMUC2ggjtuuWvxrHHHiaDH1dlQ&index=2)

## 原理剖析
一个函数，对输入的句子，预测下一个token的概率
从分布中sample不同的token来产生随机的句子

### 预训练
利用大量的文本资料，用自监督学习方法来train模型
训练得到模型就是base model
预训练后的base模型可以将语言能力泛化到多语言领域上

### fine tune
基于base model，监督训练对模型进行fine tune
在某个领域能力上对模型进行强化
让模型可以在这个方向上能力得到专精

### 强化学习
RLHF

## 引申出来的问题
- 如何使用更好的prompt
- 模型的知识时效性
- 侦测AI generate的content
- 隐私泄露问题安全问题
