# LLaMA: Open and Efficient Foundation Language Models

## abstraction
foundation models：参数范围从7B-65B，在数万亿的okens上训练的模型

## introduction
？？认为更佳的性能是通过较小模型在更多的数据上训练

## method

### pretrain dataset
- 英语CommonCrawl
- C4
- Github 
- 维基百科
- Gutenberg 和 Books3
- ArXiv
- Stack Exchange
使用字节对编码BPE算法对数据进行标记

### architecture
在transformer基础上做了部分改进
- 预测归一化[GPT3]，每个transformer子层的输入进行归一化，不是对输出归一化->RMSNorm函数
- SwiGLU激活函数[PaLM]，用SwiGLU取代ReLU，提高性能
- 旋转嵌入[GPTNeo]，删除了绝对位置嵌入，每一层添加RoPE

### optimizer
使用AdamW优化器优化

### efficient implementation
使用causal多头注意力减少内存使用，在xformers库提供

## experiments
...

## instruction fine-tunning
对指令数据进行短暂的微调可以快速改进模型的性能
