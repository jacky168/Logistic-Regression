import numpy as np
import numpy.ma as ma
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
from scipy.optimize import fmin_bfgs, fmin_cg, fmin

def sigmoid(z):
	if np.isscalar(z):
		if z > 0:
			return 1.0/(1.0+np.exp(-z))
		else:
			return np.exp(z)/(np.exp(z) + 1.0)
#	print("z.shape: " + str(z.shape))
	idx = z > 0
	out = np.zeros(z.size, dtype=np.float)
	out[idx] = 1.0/(1.0+np.exp(-z[idx]))
	out[~idx] = np.exp(z[~idx])/(np.exp(z[~idx]) + 1.0)
#	print("out.shape: " + str(out.shape))
	return out

#	return 1.0/(1.0+np.exp(-z))

def readData(fileName):
	data = np.genfromtxt(fileName, delimiter=',')
	return data

def plotData(X, y):
#one_vec = np.ones((m,1))
	pos = np.nonzero(y==1)
	neg = np.nonzero(y==0)
	scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
	scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
	title('Profits distribution')
	xlabel('Population of City in 10,000s')
	ylabel('Profit in $10,000s')
	show()

def compute_cost_reg(theta, X, y, alpha):
	J = 0.0
	epsilon = 1e-6
	np.seterr(invalid='raise')
#	print("X.shape: " + str(X.shape))
#	print("theta.shape:" + str(theta.shape))
	h = sigmoid(np.dot(X, theta))
#	h.shape = (h.size,1)
#		theta.shape = (theta.size,1)
		
	J = (-1.0/m) * (np.dot(y.T, np.log(h + epsilon)) + np.dot((1.0-y.T), np.log(1.0 - h + epsilon))) + alpha/(2*m) * np.dot(theta[1:].T, theta[1:])
#		J = (-1.0/m) * (np.dot(y.T, np.log(h)) + np.dot((1.0-y.T), np.log(1.0 - h)))
	return J

def compute_cost(theta, X, y):
	J = 0.0
	epsilon = 1e-6
	np.seterr(invalid='raise')
	h = sigmoid(np.dot(X, theta.T))
	J = (-1.0/m) * (np.dot(y.T, np.log(h + epsilon)) + np.dot((1.0-y.T), np.log(1.0 - h + epsilon)))

	return J

def compute_gradient(theta, X, y):
	m = y.size
	h = sigmoid(np.dot(X, theta.T))
	h.shape = (h.shape[0],1)
	grad = (1.0/m) * np.dot(X.T,(h-y))
#	print("grad.shape:" + str(grad.shape))
	grad.shape = (3,)
	return grad

def compute_gradient_reg(theta, X, y, alpha):
	m = y.size
	h = sigmoid(np.dot(X, theta.T))
	h.shape = (h.shape[0],1)
	temp = theta
	temp[0] = 0
	grad = np.zeros((X.shape[1],1))
	temp.shape = (theta.size,1)
#		grad = (1.0/m) * np.dot(X.T,(h-y)) + (alpha/m) * np.append([0], theta[1:])
	grad = (1.0/m) * np.dot(X.T,(h-y)) + (alpha/m) * temp
	grad.shape = (grad.size,)
	theta.shape = (theta.size,)
#	print("grad.shape:" + str(grad.shape))
#	print("theta.shape in compute_gradient_reg:" + str(theta.shape))
	return grad		


def gradient(w, X, y, alpha):
    # gradient of the logistic loss
    z = X.dot(w)
    z = sigmoid(y * z)
    z0 = (z - 1) * y
    grad = X.T.dot(z0) + alpha * w
    return grad

def predict(theta, X):
	if len(X.shape) != 1:
		m, n = X.shape
		result = np.zeros((m,1))
	else:
		n = X.shape[0]
		m = 1
		result = np.zeros((m,1))
#	print("m = " + str(m))	
#	print(X)
#	print(theta)
#	dotProduct = np.dot(X, theta)
#	print("the dot is: " + str(dotProduct))
	h = sigmoid(np.dot(X, theta))
#	print(h)
	result[np.nonzero(h > 0.5)] = 1
#	print(result)
	return result

def calculate_performance(predicted, correct):
	accuracy = correct[np.where(predicted == correct)].size / float(correct.size)
	return accuracy

def train(X, y, alpha):
	def cost(theta):
		return compute_cost_reg(theta, X, y, alpha)
#		return loss(theta, X, y, 0)

	def gradient(theta):
		return compute_gradient_reg(theta, X, y, alpha)
#		return gradient(theta, X, y, 0)
		#return compute_grad(theta, X, y)

	#Initialize theta parameters fmin_bfgs
	theta = np.zeros(X.shape[1])
#	print("theta.shape in train:" + str(theta.shape))


	# using bfgs optimization algorithm
	return fmin_bfgs(f=cost, x0 = theta, fprime=gradient, maxiter=100)
	#return fmin_cg(f=cost, x0 = theta, fprime=gradient, maxiter=100)

def gradientDescent():
		### gradient descent
	alpha = 0.001
	iters = 10000
	eps =  1e-6
	theta = np.zeros(3)
	#theta = np.array([-2.42047489e-05, 5.61832926e-03, 4.57320115e-03])
	#theta = np.random.random(3)
	#theta = np.array([-25.161272, 0.206233, 0.201470])
	for iter in xrange(iters):
		old_J = compute_cost(theta, X,y)
		grad = compute_gradient(theta, X, y)

		theta = theta - alpha * grad
		J = compute_cost(theta, X, y)

	#	diff = abs(old_J - J)
	#	print("difference %f" % diff)
		if iter % 1000 == 0:
			print("gradient:" + str(grad))
			print("cost: %f" % J)
	#	if (diff < eps):
	#		break

	print('after gradient descent...')

	J = compute_cost(theta, X, y)
	print(J)
	print(theta)


def map_feature(x1, x2):
    '''
    Maps the two input features to quadratic features.

    Returns a new feature array with more features, comprising of
    X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...

    Inputs X1, X2 must be the same size
    '''
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    out = np.ones(shape=(x1[:, 0].size, 1))

    m, n = out.shape

    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            out = np.append(out, r, axis=1)

    return out

fileName = 'ex2data2.txt'
data = readData(fileName)
(m,n) = data.shape
X = data[:,0:n-1]
X = map_feature(X[:,0],X[:,1])
y = data[:,n-1]
y.shape = (m, 1)

#plotData(X,y)

#Add intercept term to x and X_test
#one = np.ones((m,1))
#X = ma.concatenate([one,X],axis=1)
alpha = 1

theta = np.zeros(X.shape[1])
J = compute_cost_reg(theta, X, y, alpha)
#J = loss(theta, X, y, 0)
#theta = compute_gradient(theta, X, y)
#theta = gradient(theta, X, y, 0)

print("initial cost :" + str(J) + "  and initial theta :" + str(theta))

theta = train(X, y, alpha)

print("After training, cost and theta become ...")
#theta = np.array([-25.161272, 0.206233, 0.201470])
J = compute_cost_reg(theta,X, y, alpha)
print(J)
print(theta)

result = predict(theta, X)

accuracy = calculate_performance(result, y)
print(accuracy)

#X_test = np.array([1,45,85])
#prediction = predict(theta, X_test)
#print("prediction of the person with exam 1 score 45 and exam 2 score 85 is: " + str(prediction.tolist()[0][0]))
