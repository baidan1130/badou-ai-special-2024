import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#用numpy生成200个随机点
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]       #np.linspace()创建等间隔的一维数组；[:,np.newaxis]增加新维度，由（200，）变为（200，1）
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

#定义两个变量
x=tf.placeholder(tf.float32 ,[None,1])        #[None,1]表示二维矩阵，样本数不确定，样本特征为1
y=tf.placeholder(tf.float32 ,[None,1])

#定义神经网络中间层
weight_1=tf.Variable(tf.random_normal([1,10]))      #[10,1]  1个样本，10个特征
bias_1=tf.Variable(tf.zeros([1,10]))                #偏置项
sum_1=tf.matmul(x,weight_1)+bias_1
out_1=tf.nn.tanh(sum_1)                             #加入激活函数

#定义输出层
weight_2=tf.Variable(tf.random_normal([10,1]))      #[10,1]  10个样本，1个特征
bias_2=tf.Variable(tf.zeros([1,1]))                #偏置项
sum_2=tf.matmul(x,weight_2)+bias_2
out_2=tf.nn.tanh(sum_2)

#定义损失函数
loss=tf.reduce_mean(tf.square(y-out_2))
#梯度下降法定义反向传播,创建tf.train.GradientDescentOptimizer梯度下降优化器，学习率为0.1，将损失函数loss传给优化器，计算minimize（）最小值
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    prediction=sess.run(out_2,feed_dict={x:x_data})
    sess.close()

#画图
plt.figure()
plt.scatter(x_data,y_data)
plt.plot(x_data,prediction,"r-",lw=5)
plt.show()