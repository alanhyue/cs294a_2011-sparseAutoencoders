import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b

def normalizeData(patches):
    # Remove DC (mean of images)
    patches = patches - np.mean(patches)

    # Truncate to +/-3 standard deviations and scale to -1 to 1
    pstd = 3 * np.std(patches)
    patches = np.maximum(np.minimum(patches, pstd), -pstd) / pstd
    # Rescale from [-1,1] to [0.1,0.9]
    patches = (patches + 1) * 0.4 + 0.1

    return patches

def sampleIMAGES():
    """Returns 10000 patches for training"""
    from scipy.io import loadmat
    IMAGES = loadmat('IMAGES.mat')
    IMAGES = IMAGES['IMAGES']
    #  IMAGES is a 3D array containing 10 images
    #  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
    #  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
    #  it. (The contrast on these images look a bit off because they have
    #  been preprocessed using using "whitening."  See the lecture notes for
    #  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
    #  patch corresponding to the pixels in the block (21,21) to (30,30) of
    #  Image 1

    patchsize = 8 # use 8x8 patches
    numpatches = 10000

    # Initialize patches with zeros.  Your code will fill in this matrix--one
    # column per patch, 10000 columns. 
    patches = np.zeros((patchsize*patchsize, numpatches))

    # Since we are drawing 8 * 8 patches, maximum start position is 
    # 512 - 8 = 504.
    maxpos = 504

    for i in range(numpatches):
        ix_image = np.random.randint(10)
        ix_pos_start = np.random.randint(maxpos)
        block = IMAGES[ix_pos_start : ix_pos_start+patchsize, ix_pos_start : ix_pos_start+patchsize, ix_image]
        patches[:,i] = block.reshape(patchsize*patchsize)
    
    # For the autoencoder to work well we need to normalize the data
    # Specifically, since the output of the network is bounded between [0,1]
    # (due to the sigmoid activation function), we have to make sure 
    # the range of pixel values is also bounded between [0,1]
    patches = normalizeData(patches)

    return patches

def display_network(A):
    """
    This function visualizes filters in matrix A. Each column of A is a
    filter. We will reshape each column into a square image and visualizes
    on each cell of the visualization panel. 
    All other parameters are optional, usually you do not need to worry
    about it.
    """
    # rescale
    A = A - np.mean(A)

    # compute rows, cols
    L, M = A.shape
    sz = int(np.sqrt(L))
    gap = 1

    rows = cols = int(np.sqrt(M))
    while rows*cols < M: 
        rows+=1

    # initialize the picture matrix
    array = np.ones((rows*(sz+gap) + gap, cols*(sz+gap) + gap))

    # fill up the matrix with image values
    row_cnt = col_cnt = 0
    for i in range(M):
        clim = np.max(abs(A[:,i])) # for normalizing the contrast
        x, y = row_cnt*(sz+gap) + gap, col_cnt*(sz+gap) + gap
        array[x : x+sz, y : y+sz] = A[:,i].reshape((sz,sz)) / clim
        col_cnt += 1
        if col_cnt >= cols:
            row_cnt += 1
            col_cnt = 0
    plt.imshow(array, cmap='gray', interpolation='nearest')
    plt.show()

def computeNumericalGradient(J, theta):
    """
    numgrad = computeNumericalGradient(J, theta)
    theta: a vector of parameters
    J: a function that outputs a real-number. Calling y = J(theta) will return the
    function value at theta. 
    """
    m = theta.shape[0]
    # initialize numgrad with zeros
    numgrad = np.zeros(m)

    wiggle = np.zeros(m)
    e = 1e-4
    for p in range(m):
        wiggle[p] = e
        loss1, _ = J(theta - wiggle)
        loss2, _ = J(theta + wiggle)
        numgrad[p] = (loss2 - loss1) / (2 * e)
        wiggle[p] = 0
    return numgrad

def checkNumericalGradient():
    """
    This code can be used to check your numerical gradient implementation 
    in computeNumericalGradient.m
    It analytically evaluates the gradient of a very simple function called
    simpleQuadraticFunction (see below) and compares the result with your numerical
    solution. Your numerical gradient implementation is incorrect if
    your numerical solution deviates too much from the analytical solution.
    """
    def simpleQuadraticFunction(x):
        """
        this function accepts a 2D vector as input. 
        Its outputs are:
          value: h(x1, x2) = x1^2 + 3*x1*x2
          grad: A 2x1 vector that gives the partial derivatives of h with respect to x1 and x2 
        Note that when we pass @simpleQuadraticFunction(x) to computeNumericalGradients, we're assuming
        that computeNumericalGradients will use only the first returned value of this function.
        """
        value = x[0]**2 + 3*x[0]*x[1]
        grad = np.zeros(2)
        grad[0] = 2*x[0] + 3*x[1]
        grad[1] = 3*x[0]
        return value, grad

    # point at which to evaluate the function and gradient
    x = np.array([2, 88])
    
    value, grad = simpleQuadraticFunction(x)
    numgrad = computeNumericalGradient(simpleQuadraticFunction, x)

    disp = np.vstack((numgrad, grad)).T
    print(disp)
    print("The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n")

    # Evaluate the norm of the difference between two solutions.  
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    # in computeNumericalGradient.m, then diff below should be 2.1452e-12 
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print(diff)
    print('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n')

def initializeParameters(hiddenSize, visibleSize):
    """Initialize parameters randomly based on layer sizes."""
    r  = np.sqrt(6) / np.sqrt(hiddenSize+visibleSize+1)   # we'll choose weights uniformly from the interval [-r, r]
    W1 = np.random.rand(hiddenSize, visibleSize) * 2 * r - r
    W2 = np.random.rand(visibleSize, hiddenSize) * 2 * r - r

    b1 = np.zeros((hiddenSize, 1))
    b2 = np.zeros((visibleSize, 1))
    
    # Convert weights and bias gradients to the vector form.
    # This step will "unroll" (flatten and concatenate together) all 
    # your parameters into a vector, which can then be used with minFunc. 
    theta = np.hstack((W1.ravel(), W2.ravel(), b1.ravel(), b2.ravel()))
    return theta

def sparseAutoencoderCost(theta, 
        visibleSize, 
        hiddenSize, 
        lambda_, 
        sparsityParam,
        beta_,
        data):
    """
    visibleSize: the number of input units (probably 64) 
    hiddenSize: the number of hidden units (probably 25) 
    lambda_: weight decay parameter
    sparsityParam: The desired average activation for the hidden units (denoted in the lecture
                              notes by the greek alphabet rho, which looks like a lower-case "p").
    beta_: weight of sparsity penalty term
    data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
    
    The input theta is a vector (because minFunc expects the parameters to be a vector). 
    We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
    follows the notation convention of the lecture notes. 
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    # unroll parameters in theta
    W1 = theta[:hiddenSize*visibleSize].reshape(hiddenSize, visibleSize)
    W2 = theta[hiddenSize*visibleSize:2*hiddenSize*visibleSize].reshape(visibleSize, hiddenSize)
    b1 = theta[2*hiddenSize*visibleSize:2*hiddenSize*visibleSize+hiddenSize].reshape(hiddenSize,1)
    b2 = theta[2*hiddenSize*visibleSize+hiddenSize:].reshape(visibleSize,1)

    # Cost and gradient variables (your code needs to compute these values). 
    # Here, we initialize them to zeros.
    cost = 0
    W1grad = np.zeros(W1.shape)
    W2grad = np.zeros(W2.shape)
    b1grad = np.zeros(b1.shape)
    b2grad = np.zeros(b2.shape)

    # compute network cost and gradients
    m = data.shape[1]
    
    # forward pass
    A1 = data
    z2 = W1 @ A1 + b1
    A2 = sigmoid(z2)
    z3 = W2 @ A2 + b2
    A3 = sigmoid(z3)

    error = A1 - A3

    # calculate estimated activiation value, rho.
    rho = 1 / m * np.sum(A2,1).reshape(-1,1)

    # backprop with rho
    delta3 = -(A1 - A3) * A3 * (1 - A3)
    delta2 = (W2.T @ delta3 + beta_ * (-sparsityParam / rho + (1 - sparsityParam) / (1 - rho))) * A2 * (1 - A2)

    W2grad = delta3 @ A2.T
    W1grad = delta2 @ A1.T
    b2grad = np.sum(delta3,1).reshape(-1,1)
    b1grad = np.sum(delta2,1).reshape(-1,1)

    W2grad = 1/m * W2grad + lambda_ * W2
    W1grad = 1/m * W1grad + lambda_ * W1
    b2grad = 1/m * b2grad
    b1grad = 1/m * b1grad

    # compute cost and adjust costs with regularization and sparsity constriants
    mean_squared_error = 1 / m * np.sum(error**2)
    regularization_part = lambda_ / 2 * sum([np.sum(W1**2), np.sum(W2**2)])
    sparsity_part = sparsityParam * np.log(sparsityParam / rho) + (1 - sparsityParam) * np.log((1 - sparsityParam) / (1 - rho))
    cost = 0.5 * mean_squared_error + regularization_part + beta_ * np.sum(sparsity_part)

    # roll up cost and gradients to a vector format (suitable for minFunc)
    grad = np.hstack([W1grad.ravel(), W2grad.ravel(), b1grad.ravel(), b2grad.ravel()])

    return cost, grad

def train():
    ## STEP 0: set parameters of the autoencoder
    visibleSize = 8*8   #number of input units 
    hiddenSize = 25     # number of hidden units 
    sparsityParam = 0.01   # desired average activation of the hidden units.
                            # (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
                            #  in the lecture notes). 
    lambda_ = 0.0001     # weight decay parameter       
    beta_ = 3            # weight of sparsity penalty term

    ## STEP 1: sample Images
    print("sampling images...")
    patches = sampleIMAGES()
    display_network(patches[:,1:200])

    # Obtain random parameters theta
    theta = initializeParameters(hiddenSize, visibleSize)

    ## STEP 2: Implement sparseAutoencoderCost
    cost, grad = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda_,
                                        sparsityParam, beta_, patches[:,1:10])

    ## STEP 3: Gradient Checking
    # Compute gradients numerically to make sure that our implementation of gradient
    # calculation is correct.
    
    # First, let's make sure that your numerical gradient computation is correct for
    # a simple function.
    print("checking if numerical gradient computation function is implemented correctly")
    checkNumericalGradient()
    check_cost = partial(sparseAutoencoderCost,visibleSize=visibleSize,
                                        hiddenSize=hiddenSize,
                                        lambda_=lambda_,
                                        sparsityParam=sparsityParam,
                                        beta_=beta_,
                                        data=patches[:,1:10])
    numgrad = computeNumericalGradient(check_cost, theta)
    disp = np.vstack([numgrad, grad]).T
    print(disp)
    
    # Compare numerically computed gradients with the ones obtained from backpropagation
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print("the difference between numerial gradients and your gradients is {}".format(diff))
    print("the difference should be very small. Usually less than 1e-9")

    ## STEP 4: Train the sparse autoencoder with L-BFGS
    
    # randomly initialize parameters
    theta = initializeParameters(hiddenSize, visibleSize)

    partialCost = partial(sparseAutoencoderCost,visibleSize=visibleSize,
                                    hiddenSize=hiddenSize,
                                    lambda_=lambda_,
                                    sparsityParam=sparsityParam,
                                    beta_=beta_,
                                    data=patches)
    opttheta, cost, info = fmin_l_bfgs_b(partialCost, theta, 
                                            maxiter=400, disp=1)

    # print(info)

    ## STEP 5: Visualization
    W1 = opttheta[:hiddenSize*visibleSize].reshape(hiddenSize, visibleSize)
    display_network(W1.T)



if __name__ == '__main__':
    train()