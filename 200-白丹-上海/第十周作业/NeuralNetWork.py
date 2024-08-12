import numpy as np
import scipy.special
import matplotlib.pyplot as plt

class NeuralNetWork:
    def  __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        #设置输入数据
        self.i_nodes=input_nodes
        self.h_nodes=hidden_nodes
        self.o_nodes=output_nodes
        self.lr=learning_rate
        #设置权重矩阵
        #wih 输入层到中间层；who 中间层到输出层
        self.wih=np.random.rand(self.h_nodes,self.i_nodes)-0.5
        self.who=np.random.rand(self.o_nodes,self.h_nodes)-0.5     #-0.5使权重有正有负
        #定义激活函数
        self.activation_function=lambda x:scipy.special.expit(x)


    def train(self,input_list,target_list):
        #处理输入数据
        inputs=np.array(input_list,ndmin=2).T    #列向量转行向量
        targets=np.array(target_list,ndmin=2).T
        # 数据的传递
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        #计算误差，反向传递
        output_errors=targets-final_outputs
        hidden_errors=np.dot(self.who.T,output_errors*final_outputs*(1-final_outputs))

        self.who+=self.lr*np.dot((output_errors*final_outputs*(1-final_outputs)),np.transpose(hidden_outputs))
        self.wih+=self.lr*np.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)),np.transpose(inputs))





    def query(self,inputs):
        hidden_inputs=np.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)

        final_inputs=np.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)
        return final_outputs


input_nodes=784   #28×28
hidden_nodes=100
output_nodes=10
learning_rate=0.3
model=NeuralNetWork(input_nodes,hidden_nodes,output_nodes,learning_rate)

#训练过程
training_data_file=open("dataset/mnist_test.csv")
training_data_list=training_data_file.readlines()
training_data_file.close()

epoch=2
for e in range(epoch):
    for i in training_data_list:
        all_values=i.split(',')
        inputs=(np.asfarray(all_values[1:]))/255*0.99+0.01
        # 设置图片与数值的对应关系
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        model.train(inputs, targets)

#推理过程
training_data_file=open("dataset/mnist_test.csv")
training_data_list=training_data_file.readlines()
training_data_file.close()

scores=[]
for i in training_data_list:
    all_values=i.split(',')
    correct_num=int(all_values[0])
    print(f"该图片对应的数字是：{correct_num}")
    inputs=(np.asfarray(all_values[1:]))/255*0.99+0.01
    outputs=model.query(inputs)
    label=np.argmax(outputs)
    print(f"识别的结果是：{label}")
    if label == correct_num:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

#计算准确率
scores_percentage=scores.count(1)/len(scores)
print(f"准确率是：{scores_percentage}")





