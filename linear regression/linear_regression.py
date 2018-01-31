from plot import *
from cost import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

data = loaddata('./data1.txt', ',')

X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]] # constant, f1, f2
y = np.c_[data[:,2]]

plotData(data, 'Exam 1 score', 'Exam 2 score', 'Pass', 'Fail')
plt.show()
initial_theta = np.zeros(X.shape[1])
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print('Cost: \n', cost)
print('Grad: \n', grad)

res = minimize(costFunction, initial_theta, args=(X,y), jac=gradient, options={'maxiter':400})
print(res)

sigmoid(np.array([1, 45, 85]).dot(res.x.T))
plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
x1_min, x1_max = X[:,1].min(), X[:,1].max(),
x2_min, x2_max = X[:,2].min(), X[:,2].max(),

# generate meshgrid
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

# res.x contains three parameter final optimized value
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))
h = h.reshape(xx1.shape)

plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b');
plt.show()

############################################################

data2 = loaddata('data2.txt', ',')
y = np.c_[data2[:,2]]
X = data2[:,0:2]
plotData(data2, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0')

poly = PolynomialFeatures(6)
XX = poly.fit_transform(data2[:,0:2])
print(XX.shape)

initial_theta = np.zeros(XX.shape[1])
#costFunctionReg(initial_theta, 1, XX, y)

fig, axes = plt.subplots(1,3, sharey = True, figsize=(17,5))

for i, C in enumerate([0.0, 1.0, 100.0]):
    res2 = minimize(costFunctionReg, initial_theta, args=(C, XX, y), jac=gradientReg, options={'maxiter': 3000})

    accuracy = 100.0 * sum(predict(res2.x, XX) == y.ravel()) / y.size

    plotData(data2, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0', axes.flatten()[i])

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(res2.x))
    h = h.reshape(xx1.shape)
    axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');
    axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), C))