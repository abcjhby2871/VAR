# 视觉自回归模型：通过下一尺度的预测规模化图像生成
## 说明
这是南京大学2024年数学建模课程项目，我选择的是VAR这篇工作，引用方式如下。
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
所有的核心代码放在[exp.ipynb](exp.ipynb)中，包括可视化与脚本调用。我们验证了Transformer深度在16与20时脚本能够正常执行。