# Invertible Graph Neural Network iGNN
> Implementation and experiments based on the paper [Invertible Neural Network for Graph Prediction](https://arxiv.org/abs/2206.01163). The current paper is under review.

> Please direct all implementation-related inquiries to Chen Xu @ cxu310@gatech.edu.

> Citation:
```
@misc{xu2022ignn,
  doi = {10.48550/ARXIV.2206.01163},
  url = {https://arxiv.org/abs/2206.01163},
  author = {Xu, Chen and Cheng, Xiuyuan and Xie, Yao},
  keywords = {Machine Learning (stat.ML), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Invertible Neural Networks for Graph Prediction},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

```
<!-- ## Table of Contents
* [Full results](#full-results)
 -->
## Full results
- Please see [simulation.ipynb](https://github.com/hamrel-cxu/Invertible-Graph-Neural-Network-iGNN/blob/main/simulation.ipynb) regarding simulated results.
- Please see [real_data.ipynb](https://github.com/hamrel-cxu/Invertible-Graph-Neural-Network-iGNN/blob/main/real_data.ipynb) regarding real-data experiments.
- The `.gif` below visualizes how iGNN transports original densities (i.e., $X|Y$) to their corresponding ($H|Y$). The top row plots the Wasserstein-2 penalty at each block (over 40 blocks), where larger values indicate more drastic amount of transportation by the block.

<!-- ![Alt Text](https://github.com/hamrel-cxu/Invertible-Graph-Neural-Network-iGNN/blob/main/Two-moon-illustration.gif)

 -->
Two-moon Non-graph          |  3-node graph
:-------------------------:|:-------------------------:
![](https://github.com/hamrel-cxu/Invertible-Graph-Neural-Network-iGNN/blob/main/Two-moon-illustration.gif)  |  ![](https://github.com/hamrel-cxu/Invertible-Graph-Neural-Network-iGNN/blob/main/Graph-illustration.gif)
