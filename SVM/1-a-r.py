#-*- coding:utf-8 -*-
'''
@project: exuding-bert-all
@author: exuding
@time: 2019-04-23 09:59:52
'''
#svm 高斯核函数实现多分类
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()

#加载数据集，并为每类分离目标值
iris = datasets.load_iris()
#提取数据的方法
x_vals = np.array([[x[0],x[3]] for x in iris.data])

y_vals1 = np.array([1 if y==0 else -1 for y in iris.target])
y_vals2 = np.array([1 if y==1 else -1 for y in iris.target])
y_vals3 = np.array([1 if y==2 else -1 for y in iris.target])
#合并数据的方法
y_vals = np.array([y_vals1,y_vals2,y_vals3])
#数据集四个特征，只是用两个特征就可以
class1_x = [x[0] for i,x in enumerate(x_vals) if iris.target[i]==0]
class1_y = [x[1] for i,x in enumerate(x_vals) if iris.target[i]==0]
class2_x = [x[0] for i,x in enumerate(x_vals) if iris.target[i]==1]
class2_y = [x[1] for i,x in enumerate(x_vals) if iris.target[i]==1]
class3_x = [x[0] for i,x in enumerate(x_vals) if iris.target[i]==2]
class3_y = [x[1] for i,x in enumerate(x_vals) if iris.target[i]==2]
#从单类目标分类到三类目标分类，利用矩阵传播和reshape技术一次性计算所有的三类SVM，一次性计算，y_target的占位符维度是[3,None]
batch_size = 50
x_data = tf.placeholder(shape = [None,2],dtype=tf.float32)
y_target = tf.placeholder(shape=[3,None],dtype=tf.float32)
#TODO
prediction_grid = tf.placeholder(shape=[None,2],dtype=tf.float32)
b = tf.Variable(tf.random_normal(shape=[3,batch_size]))
#计算高斯核函数 TODO
gamma = tf.constant(-10.0)
dist = tf.reduce_sum(tf.square(x_data),1)
dist = tf.reshape(dist,[-1,1])
sq_dists = tf.add(tf.subtract(dist,tf.multiply(2.,tf.matmul(x_data,tf.transpose(x_data)))),tf.transpose(dist))
my_kernel = tf.exp(tf.multiply(gamma,tf.abs(sq_dists)))
#扩展矩阵维度
def reshape_matmul(mat):
    v1 = tf.expand_dims(mat,1)
    v2 = tf.reshape(v1,[3,batch_size,1])
    return (tf.matmul(v2,v1))
#计算对偶损失函数
model_output = tf.matmul(b,my_kernel)
first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b),b)
y_target_cross = reshape_matmul(y_target)
second_term = tf.reduce_sum(tf.multiply(my_kernel,tf.multiply(b_vec_cross,y_target_cross)),[1,2])
loss = tf.reduce_sum(tf.negative(tf.subtract(first_term,second_term)))
#创建预测核函数
rA = tf.reshape(tf.reduce_sum(tf.square(x_data),1),[-1,1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid),1),[-1,1])
pred_sq_dist = tf.add(tf.subtract(rA,tf.multiply(2.,tf.matmul(x_data,tf.transpose(prediction_grid)))),tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma,tf.abs(pred_sq_dist)))
#创建预测函数，这里实现的是一对多的方法，所以预测值是分类器有最大返回值的类别
prediction_output = tf.matmul(tf.multiply(y_target,b),pred_kernel)
prediction = tf.arg_max(prediction_output-tf.expand_dims(tf.reduce_mean(prediction_output,1),1),0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction,tf.arg_max(y_target,0)),tf.float32))
#准备好核函数，损失函数，预测函数以后，声明优化器函数和初始化变量
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)
init = tf.initialize_all_variables()
sess.run(init)
#该算法收敛的相当快，所以迭代训练次数不超过100次
loss_vec = []
batch_accuracy = []
for i in range(100):
    rand_index = np.random.choice(len(x_vals),size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = y_vals[:,rand_index]
    sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
    temp_loss = sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y})
    loss_vec.append(temp_loss)
    acc_temp = sess.run(accuracy,feed_dict={x_data:rand_x,y_target:rand_y,prediction_grid:rand_x})
    batch_accuracy.append(acc_temp)
    if(i+1)%25 ==0:
        print('Step #' + str(i+1))
        print('Loss #' + str(temp_loss))

x_min,x_max = x_vals[:,0].min()-1,x_vals[:,0].max()+1
y_min,y_max = x_vals[:,1].min()-1,x_vals[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,0.02),np.arange(y_min,y_max,0.02))
grid_points = np.c_[xx.ravel(),yy.ravel()]
grid_predictions = sess.run(prediction,feed_dict={x_data:rand_x,y_target:rand_y,prediction_grid:grid_points})
grid_predictions = grid_predictions.reshape(xx.shape)
#绘制训练结果，批量准确度和损失函数
#等高线图
plt.contourf(xx,yy,grid_predictions,cmap=plt.cm.Paired,alpha=0.8)
plt.plot(class1_x,class1_y,'ro',label = 'I.setosa')
plt.plot(class2_x,class2_y,'kx',label = 'I.versicolor')
plt.plot(class3_x,class3_y,'gv',label = 'T.virginica')
plt.title('Gaussian Svm Results on Iris Data')
plt.xlabel('Pedal Length')
plt.ylabel('Sepal Width')
plt.legend(loc='lower right')
plt.ylim([-0.5,3.0])
plt.xlim([3.5,8.5])
plt.show()

plt.plot(batch_accuracy,'k-',label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Sepal Width')
plt.legend(loc = 'lower right')
plt.show()

plt.plot(loss_vec,'k--')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
