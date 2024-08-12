from tensorflow.keras.datasets import mnist

(train_imgs,train_labels),(test_imgs,test_labels)=mnist.load_data()
print(f"train_imgs.shape={train_imgs.shape}")   #60000个元素，28*28
print(f"train_labels={train_labels}")
print(f"test_imgs.shape={test_imgs.shape}")     #10000个元素，28*28
print(f"tset_labels={test_labels}")

from tensorflow.keras import models
from tensorflow.keras import layers
network=models.Sequential()      #把数据层串联起来
network.add(layers.Dense(512,activation="relu",input_shape=(28*28,)))
#network.add()在已经创建的神经网络模型network中添加一个新的层
#layers.Dense(512,...)添加一个全连接层（Dense layer）,有512个神经元
#input_shape=(28,28)定义输入层是28×28的二维数组
network.add(layers.Dense(10,activation="softmax"))          #softmax:0-1之间各类别的概率分布

network.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
#network.compile()神经网络模型关键的参数与配置
#optimizer="rmsprop"：优化方式
#loss="categorical_crossentropy"：损失函数，多用于多分类问题
#metrics=["accuracy"]：准确率

train_imgs=train_imgs.reshape((60000,28*28))
train_imgs=train_imgs.astype("float32")/255
test_imgs=test_imgs.reshape((10000,28*28))
train_imgs=train_imgs.astype("float32")/255

#to_categorical:to hot编码，例如将数值7的label转化为[0,0,0,0,0,0,0,1,0,0]
from tensorflow.keras.utils import to_categorical
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

network.fit(train_imgs,train_labels,batch_size=128,epochs=5)
#模型训练

test_loss,test_acc=network.evaluate(test_imgs,test_labels,verbose=1)   #测试集评估模型
#test_loss:测试集损失值,test_acc：测试集分类准确率
print(f"test_loss={test_loss}")
print(f"test_acc={test_acc}")


res=network.predict(test_imgs)    #模型推理，返回预测值
for i in range(res[1].shape[0]):
    if (res[1][i]==1):
        print(f"the number of the picture is:{i}")
        break

print(res)

