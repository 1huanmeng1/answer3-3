import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
 
def iris_type(s):
	# python3读取数据时候，需要一个编码因此在string面前加一个b
	it = {b'Iris-setosa':0, b'Iris-versicolor':1, b'Iris-virginica':2}
	return it[s]
 
iris_feature = 'sepal length', 'sepal width', 'petal lenght', 'petal width'
 
def show_accuracy(a, b, tip):
	acc = a.ravel() == b.ravel()
	print('%s Accuracy:%.3f' %(tip, np.mean(acc)))
 
if __name__ == '__main__':
	# 加载数据
	iris_feature = 'sepal length', 'sepal width', 'petal lenght', 'petal width'
	# numpy读取
	data = np.loadtxt('date\iris.data', dtype=float, delimiter=',', converters={4:iris_type})
	x, y = np.split(data, (4,), axis=1)
	print('--------------------------')
	x = x[:, :2]
	x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6) 
	# 高斯核
	clf = svm.SVC(C=0.8, kernel='rbf', gamma=60, decision_function_shape='ovr')
	# 线性核
	# clf = svm.SVC(C=0.5, kernel='linear', decision_function_shape='ovr')
	clf.fit(x_train, y_train.ravel())
	y_hat = clf.predict(x_train)
	show_accuracy(y_hat, y_train, 'traing data')
	y_hat_test = clf.predict(x_test)
	show_accuracy(y_hat_test, y_test, 'testing data')
 
	# 开始画图
	x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
	x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
	# 生成网格采样点
	x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
	# 测试点
	grid_test = np.stack((x1.flat, x2.flat), axis=1)
	z = clf.decision_function(grid_test)
	# 预测分类值
	grid_hat = clf.predict(grid_test)
	grid_hat = grid_hat.reshape(x1.shape)
	cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
	cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
	plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
	# 样本点
	plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolor='k', s=50, cmap=cm_dark)
	# 测试点
	plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolor='none', zorder=10)
	plt.xlabel(iris_feature[0], fontsize=20)
	plt.ylabel(iris_feature[1], fontsize=20)
	plt.xlim(x1_min, x1_max)
	plt.ylim(x2_min, x2_max)
	plt.title('svm in iris data classification', fontsize=30)
	plt.grid()
	plt.show()