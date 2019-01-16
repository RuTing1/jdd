# 1.概念
+ domain： feature space； probability
+ task：label space；objective predictive function
+ source：用于训练模型的域/任务
+ target：用source的模型对自己的数据进行预测/分类/聚类等机器学习任务的域/任务

# 2.传统机器学习的假设
+ 相同的特征空间
+ 相同的数据分布

# 3.什么时候需要用到迁移学习？
+ 迁移学习的边界在哪里：conditional Kolmogorov complexity去衡量tasks之间的相关性

# 4.迁移学习分类
+ (1) Instance-based TL（样本迁移）<br/>
  instance reweighting（样本重新调整权重） <br/>
  importance sampling（重要性采样） <br/>
+ (2) Feature-representation-transfer（特征迁移）
+ (3) Parameter-transfer（参数/模型迁移）
+ (4) Relational-knowledge-transfer（关系迁移）

# 5.目前的主要应用
+ 提高feature space相同probability不同的表现
+ feature space不同的domain和task进行迁移
+ 主要应用在小且波动不大的数据集

# 6.目前应用中的问题
