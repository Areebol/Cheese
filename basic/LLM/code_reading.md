<!--
 * @Descripttion: 
 * @version: 1.0
 * @Author: Areebol
 * @Date: 2023-06-10 21:32:48
-->

| 内容               | 代码                                                                                                                                                    | 视频                                                |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| 开源语言大模型代码 | [LLaMA](https://github.com/huggingface/transformers/blob/ba695c1efd55091e394eb59c90fb33ac3f9f0d41/src/transformers/models/llama/modeling_llama.py#L612) | [视频](https://www.bilibili.com/video/BV1pu411o7BE) |
| 开源语言大模型代码 | [OPT](https://github.com/huggingface/transformers/blob/ba695c1efd55091e394eb59c90fb33ac3f9f0d41/src/transformers/models/opt/modeling_opt.py#L819) | * |
| 开源语言大模型代码 | [BLOOM](https://github.com/huggingface/transformers/blob/ba695c1efd55091e394eb59c90fb33ac3f9f0d41/src/transformers/models/bloom/modeling_bloom.py#L828) | * |




"代码2：https://github.com/ymcui/Chinese-LLaMA-Alpaca 
这一代码实现了语言大模型的继续预训练，整个训练流程包括词表扩充、预训练和指令精调三部分：
1）词表扩充：原版LLaMA对中文的支持非常有限，这一代码在原版LLaMA的基础上进一步扩充了中文词表；
2）预训练：先冻结transformer参数，仅训练embedding；然后训练embedding的同时也更新模型整体参数；
3）指令精调：这一步骤和InstructGPT的SFT一致，可以和代码3中的Supervised Finetuning阶段一起理解；
关于这3个部分的更多细节可以查看https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E8%AE%AD%E7%BB%83%E7%BB%86%E8%8A%82 



"代码3：https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat 
这一代码实现了InstructGPT（ChatGPT的核心论文）中的3个步骤，包括
1）Supervised Finetuning阶段，见https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning 
2）Reward Model Training阶段，见https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning 
3）RLHF finetuning阶段，见https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning

阅读代码之前，可以观看下列视频学习基础知识：
https://www.bilibili.com/video/BV1hd4y187CR
https://www.bilibili.com/video/BV1ts4y1T7UH
