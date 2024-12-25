# 视觉自回归模型：通过下一尺度的预测规模化图像生成
## 说明
这是南京大学2024年数学建模课程项目，我选择的是 [Visual AutoRegressive modeling](https://github.com/FoundationVision/VAR) 这篇工作，引用方式如下。
```
@misc{tian2024visualautoregressivemodelingscalable,
      title={Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction}, 
      author={Keyu Tian and Yi Jiang and Zehuan Yuan and Bingyue Peng and Liwei Wang},
      year={2024},
      eprint={2404.02905},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.02905}, 
}
```
## 安装
```
pip install jupyter
```
## 使用
所有的核心代码放在[exp.ipynb](exp.ipynb)中，包括可视化与脚本调用。我们验证了Transformer深度在16与20时脚本能够正常执行。额外的三个脚本的作用如下：
1. [sample.py](sample.py): 提供了条件图像生成的代码。
2. [scale.py](scale.py):提供了尺度定律验证的重构损失与错误率计算。
3. [auto.py](auto.py):提供了图像质量评估的调用代码。