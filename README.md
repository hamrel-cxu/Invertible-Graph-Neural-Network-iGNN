# Invertible Graph Neural Network iGNN
> Implementation and experiments based on the paper "Invertible Neural Network for Graph Prediction". The current paper is under review by IEEE Journal on Selected Areas in Information Theory—Special Issue on Deep Learning Methods for Inverse Problems. 

> Please direct all implementation-related inquiries to Chen Xu @ cxu310@gatech.edu.

> Citation:
```

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
