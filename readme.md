# 前沿
之前做了个博弈类的强化学习，奈何游戏选的不好，游戏机制的漏洞很快被网络捕获，导致没法深入研究，此次就做一个单独和环境交互的游戏，来进行强化学习的一些探索。

# 游戏环境
+ game:2048
+ step: ↑ ↓ ← → ，return s,r,done,s'
+ r:每回合得分，死掉扣1000

# 模型测试
## 随机版本
+ 随机进行1000次,得分最高2500左右，结局大致如下，还是比较菜的一个版本。
    2	 4	    16	    2

    32	 128	64	    8

    16	 2	    16	    2

    4	 8	    256	    8

## DQN2013版
Q1: 结果取向与走不动
A1:
+ 训练集剔除走不动的s-s'，这样容易让模型趋于走不动的地步。

Q1：对于初始学习能力较弱，4*4直接喂给dnn,且初始数据不足的情况下，没有能学习到尽可能去合并的概念。
A1：
+ 尝试增强特征提取，行列分开处理。
+ 尝试优化reward，尽可能保留更多的空格

# 总结
+ 对于2048来讲，是个模型已知的游戏，为何得出此结论？
    + P（s'|s,a） 是已知的一个随机事件，可模拟。
    + R（s'|s,a）是已知
    + 整个环境是可模拟的，那么也就可以通过值迭代和策略迭代的方法来完成，但是有16个位置，每个位置算10种可能，16^10的可能性也太多了，9^9也太多了，4^4还可以接受，那么我们
    还是尝试从2*2的方格开始解决这个问题。

# 效果分析
+ 网络：整体网络上，采用emb,ont-hot方式最后都拿到了不错的结果，可以出2048，原生矩阵的没有到2048，可能是步数不够，不过上个PPT里面用raw拿到了4096，证明其有效性。
+ 网络结构：dqn2013也能拿到2048甚至4096，好像拟合的更快。ddqn也拿到了最后结果，但是dueling，上述ppt展示了，更佳的效果性能。自己没有尝试。
+ 学习率：学习率0.0005，离散衰减，这个学习率还挺重要的。
+ e-greedy：这个一定要做衰减，不然后面512后难收敛提升。
+ memory:最开始设置了100w，后来看别人之后，改成了6000，这里应该影响比较大，100W的时候特别容易过估计。
+ reward设置：设置为分数，做个log2即可。
+ 使用纯mcts的话，200次模拟，几乎很容易达到2048，就是很慢。如果想用mcts给做训练数据，这个优化得做好，不然感觉速度有点慢的无法接受。（alphaGo反正就是这么整出来的

# 关键资料
+ https://cs.uwaterloo.ca/~mli/zalevine-dqn-2048.pdf
+ https://github.com/navjindervirdee/2048-deep-reinforcement-learning.git

