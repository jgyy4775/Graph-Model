# Graph-Model
This is a collection of resources related with graph neural networks.


## Contents

- [Survey papers](#surveypapers)
- [Papers](#papers)
  - [Recuurent Graph Neural Networks](#rgnn)
  - [Convolutional Graph Neural Networks](#cgnn)
  - [Spatial-Temporal Graph Neural Networks](#stgnn)
  - [Application](#application)
     - [Computer Vision](#cv)
     - [Natural Language Processing](#nlp)
- [Library](#library)
<a name="surveypapers" />

## Survey papers

1. **A Comprehensive Survey on Graph Neural Networks.** *Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, Philip S. Yu.* 2019 [paper](https://arxiv.org/pdf/1901.00596.pdf)

2. **Relational inductive biases, deep learning, and graph networks.**
*Peter W. Battaglia, Jessica B. Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, Caglar Gulcehre, Francis Song, Andrew Ballard, Justin Gilmer, George Dahl, Ashish Vaswani, Kelsey Allen, Charles Nash, Victoria Langston, Chris Dyer, Nicolas Heess, Daan Wierstra, Pushmeet Kohli, Matt Botvinick, Oriol Vinyals, Yujia Li, Razvan Pascanu.* 2018. [paper](https://arxiv.org/pdf/1806.01261.pdf)

3. **Attention models in graphs.** *John Boaz Lee, Ryan A. Rossi, Sungchul Kim, Nesreen K. Ahmed, Eunyee Koh.* 2018. [paper](https://arxiv.org/pdf/1807.07984.pdf)

4. **Deep learning on graphs: A survey.** Ziwei Zhang, Peng Cui and Wenwu Zhu. 2018. [paper](https://arxiv.org/pdf/1812.04202.pdf)

5. **Graph Neural Networks: A Review of Methods and Applications** *Jie Zhou, Ganqu Cui, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, Maosong Sun.* 2018 [paper](https://arxiv.org/pdf/1812.08434.pdf)



<a name="papers" />

## Papers

<a name="rgnn" />

## Recurrent Graph Neural Networks

1. **Gated graph sequence neural networks.** *Yujia Li, Richard Zemel, Marc Brockschmidt, Daniel Tarlow.* ICLR 2015. [paper](https://arxiv.org/pdf/1511.05493.pdf)

2. **Learning steady-states of iterative algorithms over graphs.** *Hanjun Dai, Zornitsa Kozareva, Bo Dai, Alexander J. Smola, Le Song* ICML 2018. [paper](http://proceedings.mlr.press/v80/dai18a/dai18a.pdf)

3. **GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models** *Jiaxuan You, Rex Ying, Xiang Ren, William L. Hamilton, Jure Leskovec*  [paper](https://arxiv.org/abs/1802.08773) [code](https://github.com/JiaxuanYou/graph-generation)

<a name="cgnn" />

## Convolutional Graph Neural Networks

### Spectral

1. **Semi-supervised classification with graph convolutional networks.** *Thomas N. Kipf, Max Welling.* ICLR 2017. [paper](https://arxiv.org/pdf/1609.02907.pdf)

1. **Cayleynets: graph convolutional neural networks with complex rational spectral filters.** *Ron Levie, Federico Monti, Xavier Bresson, Michael M. Bronstein.* 2017. [paper](https://arxiv.org/pdf/1705.07664.pdf)

1. **Simplifying Graph Convolutional Networks.** *Felix Wu, Tianyi Zhang, Amauri Holanda de Souza Jr., Christopher Fifty, Tao Yu, Kilian Q. Weinberger.* ICML 2019. [paper](https://arxiv.org/pdf/1902.07153.pdf) [code](https://github.com/Tiiiger/SGC)

1. **Graph Wavelet Neural Network.** *Bingbing Xu, Huawei Shen, Qi Cao, Yunqi Qiu, Xueqi Cheng.* ICLR 2019. [paper](https://openreview.net/pdf?id=H1ewdiR5tQ)


1. **DIFFUSION SCATTERING TRANSFORMS ON GRAPHS.** *Fernando Gama, Alejandro Ribeiro, Joan Bruna.* ICLR 2019. [paper](https://arxiv.org/pdf/1806.08829.pdf)

### Spatial

1. **Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs** *Martin Simonovsky, Nikos Komodakis* CVPR 2017. [paper](https://arxiv.org/pdf/1704.02901.pdf) 

1. **Geometric deep learning on graphs and manifolds using mixture model cnns.** *Federico Monti, Davide Boscaini, Jonathan Masci, Emanuele Rodolà, Jan Svoboda, Michael M. Bronstein.* CVPR 2017. [paper](https://arxiv.org/pdf/1611.08402.pdf)


1. **Robust spatial filtering with graph convolutional neural networks.** 2017. *Felipe Petroski Such, Shagan Sah, Miguel Dominguez, Suhas Pillai, Chao Zhang, Andrew Michael, Nathan Cahill, Raymond Ptucha.* [paper](https://arxiv.org/abs/1703.00792)


1. **Structure-Aware Convolutional Neural Networks.** *Jianlong Chang, Jie Gu, Lingfeng Wang, Gaofeng Meng, Shiming Xiang, Chunhong Pan.* NeurIPS 2018. [paper](http://papers.nips.cc/paper/7287-structure-aware-convolutional-neural-networks.pdf) [code](https://github.com/vector-1127/SACNNs)


1. **On filter size in graph convolutional network.** *D. V. Tran, A. Sperduti et al.* SSCI. IEEE, 2018. [paper](https://arxiv.org/pdf/1811.10435.pdf)

1. **Predict then Propagate: Graph Neural Networks meet Personalized PageRank.** *Johannes Klicpera, Aleksandar Bojchevski, Stephan Günnemann.* ICLR 2019. [paper](https://arxiv.org/pdf/1810.05997.pdf) [code](https://github.com/benedekrozemberczki/APPNP)


#### Attention/Gating Mechanisms 

1. **Graph Attention Networks.**
*Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio.* ICLR 2018. [paper](https://arxiv.org/pdf/1710.10903.pdf) [code](https://github.com/PetarV-/GAT)

1. **Gaan: Gated attention networks for learning on large and spatiotemporal graphs.** *Jiani Zhang, Xingjian Shi, Junyuan Xie, Hao Ma, Irwin King, Dit-Yan Yeung.* 2018. [paper](https://arxiv.org/pdf/1803.07294.pdf)

1. **Geniepath: Graph neural networks with adaptive receptive paths.** Ziqi Liu, Chaochao Chen, Longfei Li, Jun Zhou, Xiaolong Li, Le Song, Yuan Qi. AAAI 2019. [paper](https://arxiv.org/pdf/1802.00910.pdf)

1. **Graph Representation Learning via Hard and Channel-Wise Attention Networks.** *Hongyang Gao, Shuiwang Ji.* 2019 KDD. [paper](https://www.kdd.org/kdd2019/accepted-papers/view/graph-representation-learning-via-hard-and-channel-wise-attention-networks) 

1. **Understanding Attention and Generalization in Graph Neural Networks.** *Boris Knyazev, Graham W. Taylor, Mohamed R. Amer.* NeurIPS 2019. [paper](https://arxiv.org/abs/1905.02850)

#### Convolution 

1. **Learning convolutional neural networks for graphs.** *Mathias Niepert, Mohamed Ahmed, Konstantin Kutzkov.* ICML 2016. [paper](https://arxiv.org/pdf/1605.05273.pdf)

1. **Large-Scale Learnable Graph Convolutional Networks.** *Hongyang Gao, Zhengyang Wang, Shuiwang Ji.* KDD 2018. [paper](https://arxiv.org/pdf/1808.03965.pdf)


#### Training Methods

1. **Adaptive Sampling Towards Fast Graph Representation Learning.** *Wenbing Huang, Tong Zhang, Yu Rong, Junzhou Huang.* NeurIPS 2018. [paper](https://arxiv.org/pdf/1809.05343.pdf) [code]()

1. **FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling.**
*Jie Chen, Tengfei Ma, Cao Xiao.* ICLR 2018. [paper](https://arxiv.org/pdf/1801.10247.pdf)

1. **Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks.** KDD 2019. [paper](https://arxiv.org/pdf/1905.07953.pdf) [code](https://github.com/google-research/google-research/tree/master/cluster_gcn)


#### Graph Classfication

1. **Contextual graph markov model: A deep and generative approach to graph processing.** *D. Bacciu, F. Errica,  A. Micheli.* ICML 2018. [paper](https://arxiv.org/abs/1805.10636)

1. **Adaptive graph convolutional neural networks.** *Ruoyu Li, Sheng Wang, Feiyun Zhu, Junzhou Huang.* AAAI 2018. [paper](https://arxiv.org/pdf/1801.03226.pdf)


1. **Graph capsule convolutional neural networks** *Saurabh Verma, Zhi-Li Zhang.* 2018. [paper](https://arxiv.org/abs/1805.08090)


1. **Capsule Graph Neural Network** *Zhang Xinyi, Lihui Chen.* ICLR 2019. [paper](https://openreview.net/pdf?id=Byl8BnRcYm)


#### Analysis

1. **Deeper insights into graph convolutional networks for semi-supervised learning.** *Qimai Li, Zhichao Han, Xiao-Ming Wu.* AAAI 2018. [paper](https://arxiv.org/pdf/1801.07606.pdf)

1. **How powerful are graph neural networks?** *Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka.* ICLR 2019. [paper](https://arxiv.org/pdf/1810.00826.pdf)

1. **Can GCNs Go as Deep as CNNs?.** *Guohao Li, Matthias Müller, Ali Thabet, Bernard Ghanem.* 2019. ICCV 2019. [paper](https://arxiv.org/abs/1904.03751)

1. **Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks.** *Christopher Morris, Martin Ritzert, Matthias Fey, William L. Hamilton, Jan Eric Lenssen, Gaurav Rattan, Martin Grohe* AAAI 2019. [paper](https://arxiv.org/pdf/1810.02244.pdf)


<a name="stgnn" />

## Spatial-Temporal Graph Neural Networks

1. **Structured sequence modeling with graph convolutional recurrent networks.** *Youngjoo Seo, Michaël Defferrard, Pierre Vandergheynst, Xavier Bresson.* 2016. [paper](https://arxiv.org/pdf/1612.07659.pdf)

1. **Structural-rnn: Deep learning on spatio-temporal graphs.** *Ashesh Jain, Amir R. Zamir, Silvio Savarese, Ashutosh Saxena.* CVPR 2016. [paper](https://arxiv.org/abs/1511.05298)


1. **Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs.** *
Rakshit Trivedi, Hanjun Dai, Yichen Wang, Le Song.* ICML 2017 [paper](https://arxiv.org/pdf/1705.05742.pdf)

1. **Deep multi-view spatial-temporal network for taxi.** *Huaxiu Yao, Fei Wu, Jintao Ke, Xianfeng Tang, Yitian Jia, Siyu Lu, Pinghua Gong, Jieping Ye, Zhenhui Li.* AAAI 2018. [paper](https://arxiv.org/abs/1802.08714)

1. **Spatial temporal graph convolutional networks for skeleton-based action recognition.** *Sijie Yan, Yuanjun Xiong, Dahua Lin.* AAAI 2018. [paper](https://arxiv.org/abs/1801.07455)


1. **Diffusion convolutional recurrent neural network: Data-driven traffic forecasting.** *Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu.* ICLR 2018. [paper](https://arxiv.org/pdf/1707.01926.pdf)

1. **Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting.** *Bing Yu, Haoteng Yin, Zhanxing Zhu.* IJCAI 2018. [paper](https://arxiv.org/pdf/1709.04875.pdf)

1. **Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting.** *Shengnan Guo, Youfang Lin, Ning Feng, Chao Song, HuaiyuWan* AAAI 2019. [paper](https://aaai.org/ojs/index.php/AAAI/article/view/3881)

1. **Spatio-Temporal Graph Routing for Skeleton-based Action Recognition.** *Bin Li, Xi Li, Zhongfei Zhang, Fei Wu.*  AAAI 2019. [paper](https://www.aaai.org/Papers/AAAI/2019/AAAI-LiBin.6992.pdf)

1. **Graph wavenet for deep spatial-temporal graph modeling** *Z. Wu, S. Pan, G. Long, J. Jiang, and C. Zhang* IJCAI 2019. [paper](https://arxiv.org/abs/1906.00121)

1. **Semi-Supervised Hierarchical Recurrent Graph Neural Network for City-Wide Parking Availability Prediction.** *Weijia Zhang, Hao Liu, Yanchi Liu, Jingbo Zhou, Hui Xiong.* AAAI 2020. [paper](https://arxiv.org/pdf/1911.10516.pdf)

1. **Temporal Graph Networks for Deep Learning on Dynamic Graphs** *Emanuele Rossi, Ben Chamberlain, Fabrizio Frasca, Davide Eynard, Federico Monti, Michael Bronstein.* 2020.  [paper](https://arxiv.org/abs/2006.10637) [code](https://github.com/twitter-research/tgn)
<a name="application" />

## Application

<a name="cv" />

### Computer Vision

1. **Syncspeccnn: Synchronized spectral cnn for 3d shape segmentation.** *Li Yi, Hao Su, Xingwen Guo, Leonidas Guibas.* CVPR 2017. [paper](https://arxiv.org/pdf/1612.00606.pdf)

1. **A simple neural network module for relational reasoning.** *Adam Santoro, David Raposo, David G.T. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, Timothy Lillicrap.* NIPS 2017 [paper](https://arxiv.org/pdf/1706.01427.pdf)

1. **Situation Recognition with Graph Neural Networks.** *Ruiyu Li, Makarand Tapaswi, Renjie Liao, Jiaya Jia, Raquel Urtasun, Sanja Fidler.* ICCV 2017. [paper](https://arxiv.org/pdf/1708.04320)

1. **Image generation from scene graphs.** *Justin Johnson, Agrim Gupta, Li Fei-Fei.* CVPR 2018. [paper](https://arxiv.org/pdf/1804.01622.pdf)

1. **PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.**
*Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas.* CVPR 2018. [paper](https://arxiv.org/pdf/1612.00593.pdf)

1. **Iterative visual reasoning beyond convolutions.** *Xinlei Chen, Li-Jia Li, Li Fei-Fei, Abhinav Gupta.* CVPR 2018. [paper](https://arxiv.org/pdf/1803.11189.pdf)

1. **Large-scale point cloud semantic segmentation with superpoint graphs.** *Loic Landrieu, Martin Simonovsky.* CVPR 2018. [paper](https://arxiv.org/pdf/1711.09869.pdf)


1. **Learning Conditioned Graph Structures for Interpretable Visual Question Answering.**
*Will Norcliffe-Brown, Efstathios Vafeias, Sarah Parisot.* NeurIPS 2018. [paper](https://arxiv.org/pdf/1806.07243)

1. **Out of the box: Reasoning with graph convolution nets for factual visual question answering.** *Medhini Narasimhan, Svetlana Lazebnik, Alexander G. Schwing.* NeurIPS 2018. [paper](https://arxiv.org/pdf/1811.00538.pdf)

1. **Symbolic Graph Reasoning Meets Convolutions.** *Xiaodan Liang, Zhiting Hu, Hao Zhang, Liang Lin, Eric P. Xing.* NeurIPS 2018. [paper](http://papers.nips.cc/paper/7456-symbolic-graph-reasoning-meets-convolutions.pdf)

1. **Few-shot learning with graph neural networks.** *Victor Garcia, Joan Bruna.* ICLR 2018. [paper](https://arxiv.org/abs/1711.04043)

1. **Factorizable net: an efficient subgraph-based framework for scene graph generation.** *Yikang Li, Wanli Ouyang, Bolei Zhou, Jianping Shi, Chao Zhang, Xiaogang Wang.* ECCV 2018. [paper](https://arxiv.org/abs/1806.11538)

1. **Graph r-cnn for scene graph generation.** *Jianwei Yang, Jiasen Lu, Stefan Lee, Dhruv Batra, Devi Parikh.* ECCV 2018. [paper](https://arxiv.org/pdf/1808.00191.pdf)

1. **Learning Human-Object Interactions by Graph Parsing Neural Networks.** *Siyuan Qi, Wenguan Wang, Baoxiong Jia, Jianbing Shen, Song-Chun Zhu.* ECCV 2018. [paper](https://arxiv.org/pdf/1808.07962.pdf)

1. **Neural graph matching networks for fewshot 3d action recognition.** *Michelle Guo, Edward Chou, De-An Huang, Shuran Song, Serena Yeung, Li Fei-Fei* ECCV 2018. [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Michelle_Guo_Neural_Graph_Matching_ECCV_2018_paper.pdf)

1. **Rgcnn: Regularized graph cnn for point cloud segmentation.** *Gusi Te, Wei Hu, Zongming Guo, Amin Zheng.* 2018. [paper](https://arxiv.org/pdf/1806.02952.pdf)

1. **Dynamic graph cnn for learning on point clouds.** *Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon.* 2018. [paper](https://arxiv.org/pdf/1801.07829.pdf)

<a name="nlp" />

### Natural Language Processing
1. **Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling.**
*Diego Marcheggiani, Ivan Titov.* EMNLP 2017. [paper](https://arxiv.org/abs/1703.04826)

1. **Graph Convolutional Encoders for Syntax-aware Neural Machine Translation.**
*Joost Bastings, Ivan Titov, Wilker Aziz, Diego Marcheggiani, Khalil Sima'an.* EMNLP 2017. [paper](https://arxiv.org/pdf/1704.04675)



1. **Diffusion maps for textual network embedding.** *Xinyuan Zhang, Yitong Li, Dinghan Shen, Lawrence Carin.* NeurIPS 2018. [paper](https://arxiv.org/pdf/1805.09906.pdf)

1. **A Graph-to-Sequence Model for AMR-to-Text Generation.**
*Linfeng Song, Yue Zhang, Zhiguo Wang, Daniel Gildea.* ACL 2018. [paper](https://arxiv.org/abs/1805.02473)

1. **Graph-to-Sequence Learning using Gated Graph Neural Networks.** *Daniel Beck, Gholamreza Haffari, Trevor Cohn.* ACL 2018. [paper](https://arxiv.org/pdf/1806.09835.pdf)

1. **Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks.** *Zhichun Wang, Qingsong Lv, Xiaohan Lan, Yu Zhang.* EMNLP 2018. [paper](http://www.aclweb.org/anthology/D18-1032)

1. **Graph Convolution over Pruned Dependency Trees Improves Relation Extraction.**  *Yuhao Zhang, Peng Qi, Christopher D. Manning.* EMNLP 2018. [paper](https://arxiv.org/pdf/1809.10185)

1. **Multiple Events Extraction via Attention-based Graph Information Aggregation.** *Xiao Liu, Zhunchen Luo, Heyan Huang.* EMNLP 2018. [paper](https://arxiv.org/pdf/1809.09078.pdf)

1. **Exploiting Semantics in Neural Machine Translation with Graph Convolutional Networks.** *Diego Marcheggiani, Joost Bastings, Ivan Titov.* NAACL 2018. [paper](http://www.aclweb.org/anthology/N18-2078)

1. **Graph Convolutional Networks for Text Classification.** *Liang Yao, Chengsheng Mao, Yuan Luo.* AAAI 2019. [paper](https://arxiv.org/pdf/1809.05679.pdf)
<a name="web" />


## Library

1. [pytorch geometric](https://github.com/rusty1s/pytorch_geometric)

1. [deep graph library](https://github.com/dmlc/dgl)

1. [graph nets library](https://github.com/deepmind/graph_nets)

1. [GNN-based Fraud Detection Toolbox](https://github.com/safe-graph/DGFraud)
