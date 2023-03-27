# Invertible Graph Neural Network iGNN
> Implementation and experiments based on the paper [Invertible Neural Network for Graph Prediction](https://arxiv.org/abs/2206.01163), accepted at the IEEE Journal on Selected Areas in Information Theory---Deep Learning for Inverse Problems.

> Citation:
```
@ARTICLE{9950057,
  author={Xu, Chen and Cheng, Xiuyuan and Xie, Yao},
  journal={IEEE Journal on Selected Areas in Information Theory}, 
  title={Invertible Neural Networks for Graph Prediction}, 
  year={2022},
  volume={3},
  number={3},
  pages={454-467},
  doi={10.1109/JSAIT.2022.3221864}}
```

- Please see [example.ipynb](https://github.com/hamrel-cxu/Invertible-Graph-Neural-Network-iGNN/blob/main/example.ipynb) regarding how to use the method.
- The movie below visualizes how iGNN transports original densities $X|Y$ of the three-moon dataset to their corresponding $H|Y$. The top row plots the Wasserstein-2 penalty at each block, where larger values indicate more drastic amount of movement by the block.

<p align="center">
  <img src="https://github.com/hamrel-cxu/Invertible-Graph-Neural-Network-iGNN/blob/main/Three_moon.gif" width="350" height="450" />
</p>
