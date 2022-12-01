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

- Please see [simulation.ipynb](https://github.com/hamrel-cxu/Invertible-Graph-Neural-Network-iGNN/blob/main/simulation.ipynb) regarding simulated results.
- The movie below visualizes how iGNN transports original densities (i.e., $X|Y$) to their corresponding ($H|Y$). The top row plots the Wasserstein-2 penalty at each block, where larger values indicate more drastic amount of transportation by the block.

Three moons        
:-------------------------:
![](https://github.com/hamrel-cxu/Invertible-Graph-Neural-Network-iGNN/blob/main/Three_moon.mp4) 
