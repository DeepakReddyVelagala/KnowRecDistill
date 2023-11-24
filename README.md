# KnowRecDistill

Repository containing knowledge distillation for Recommender systems
Work done by Deepak(Me), Tapishi, Safder, Waris as a part of the course Deep learning.

##  Motivation for this work
Recommender systems have become an essential part of many online platforms, such as Netflix, Amazon, Youtube. The increasing importance of recommender systems has led to a growing need for efficient recommender systems. Efficient not only in terms of the accuracy but also in terms of scalability, accessibility and other factors. This can be seen clearly from the celebrated Netflix prize, where the prize-winning model (ensemble of around 160 models) despite its remarkable accuracy, remained too large and intricate for practical deployment nevertheless the event has made great contributions to the development in field of recommender systems.

Meanwhile the rise of edge computing has led to active research on tiny-ML, which aims to build small but efficient machine learning systems, one of such many interesting methods is Knowledge distillation.

Our project aims study and understand the methods currently in use for balancing accuracy and other factors for efficiency by harnessing the concept of knowledge distillation. We endeavor to distill the extensive wisdom embedded within a large, intricate teacher model, such as the state-of-the-art recommender system, into a sleeker and faster student model.

## Brief Intro to Recommender systems

Recommender systems learn about people's likes and dislikes, past choices, and other features by looking at how they interact with products, such as by viewing them, clicking on them, liking them, or buying them.

These systems can be classified into three main categories based on the primary techniques and data sources they use to make recommendations.

- **Collaborative Filtering**: Collaborative filtering methods are founded on the idea that users who have shown similar behaviours or preferences in the past will likely have similar preferences in the future. This approach relies on user-item interactions to make recommendations. There are two primary types of collaborative filtering: user-based and item-based.

- **Content-Based Filtering**: Content-based filtering recommends items to users based on the attributes of the items and a profile of the user's preferences. This approach relies on extracting features or keywords from items and matching them to a user profile.

- **Hybrid Recommender Systems**: Hybrid systems combine collaborative and content-based filtering methods to leverage the strengths of both. These systems aim to improve recommendation accuracy and overcome some of the limitations of individual approaches

Recommender systems are employed to address a wide array of specific problems and challenges in various domains, such as Click-Through Rate (CTR) Prediction, Next-Item Recommendation, Item Ranking,  Cold-Start Problem,  Session-Based Recommendations, Multi-Armed Bandit Problems, Cross-Domain Recommendations etc...

### Some examples of popular Recommender systems are as follows
- Neural Collaborative Filtering 
- Wide and Deep Learning for Recommender System
- DeepFM
- BERT4Rec
- LightGNN etc..


## Brief take on Knowledge Distillation
Knowledge Distillation involves the transfer of knowledge from one or more large, complex models to a single, more compact model that can be efficiently deployed within practical real-world limitations. A knowledge distillation setup comprises three fundamental elements: the knowledge itself, the distillation algorithm, and the teacher-student architecture.

## Knowledge Distillation for Recommender Systems

We have studied various frameworks for knowledge distillation of different types of Recommender system.

Following are a few of this:
 - A novel Enhanced Collaborative Autoencoder with knowledge distillation for top-N recommender systems. Pan, Yiteng et al. Neurocomputing 201
 - Scene-adaptive Knowledge Distillation for Sequen-
tial Recommendation via Differentiable Architecture
Search (2021) Lei chen et al
 - DE-RRD: A Knowledge Distilla-
tion Framework for Recommender Systems by SeongKu Kang, Junyoung Hwang, and Hwanjo Yu

etc..

## Topological Distillation
We studied the paper by "Topology Distillation for Recommender System
SeongKu Kang, Junyoung Hwang, Wonbin Kweon, Hwanjo Yu".

- Using this paper as reference we tried to grasp the concepts of Topological Distillation.
- This paper propose a novel method named Hierarchical Topology Distillation (HTD) which distills the topology
hierarchically to cope with the large capacity gap between student and teacher model.
- Our main is aim to study and Implement Topological distillation refered as Fully topological distillation(FTD) in paper, on Deep learning based recommender systems.
- And Compare FTD and HTD using a toy model.

## Code and Experiments

- We Compared the HTD vs FTD method using BPR (a prominent Matrix Factorization based recommender system) and CiteUlike dataset for top N recommendations.
- We are able clearly observe the conclusions made in TD for RecSys paper. Where Student model of smaller capacity or size performs well with knowledge distillation using HTD method and student model of significant size performs better when trained with FTD method.
  
*We credit authors Topological distillation for recommender systems paper for the code in the HTD vs FTD notebook. We used their code and modified according to our study.*

 - To study further on Full Topological distillation, we implemented a Deep learning based Top n recommendation system, Neural Collaborative filtering.
 - Used the famous recommender systems Dataset Movielens 100k for training.
 - We also implemented the Full topological distillation method for the Neural Collaborative filtering model and Trained, experimented on the student models with less layers and smaller embeddings compared to parent model.

### Explanation various files in the code base

All the models are implemented using pytorch
 - *dataset* : folder containing csv's from the movie lens 100k dataset and train, val, test csvs are generated by us using the preprocess.py.

The preprocessing is done to accomodate it for training Top n recommendation system.
We assumed if a user rated a film in some way it is more interesting to user compared to unrated films.
 So we marked rated films as 1 and unrated as negative, 0.


 - *src* : folder containing most of the code
 > - *rec_with_{model}.py* : use these scripts to select any valid user (say) and print the movies recommended to by that model.
 > -  *model.py* : contains the architecture implementation of the models
 > - *train_{model}* : contains the code to train the models.
 - *save_models* : contains trained models
 - *KnowRecDistill.ipynb* : contains all code in src used for training using colab gpu's (might be bit messy ;)


