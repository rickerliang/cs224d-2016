import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def ce(Yt, Yhat):
    cost = Yt * np.log(Yhat)
    cost = -np.sum(cost, axis=1)

    grad = - (Yt / Yhat)

    return cost



def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    print "-------params"
    print params
    print "-------params"
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    #print data.shape

    print "-----------"
    print labels.shape
    sigmoid_out = sigmoid(data.dot(W1) + b1)
    softmax_output = softmax(sigmoid_out.dot(W2) + b2)
    cost = ce(labels, softmax_output).sum()
    print cost.shape
    print "==========="
    ### END YOUR CODE


    ### YOUR CODE HERE: backward propagation

    print ">>>>>>>>>>>>"
    print "W1"
    print W1.shape
    print "b1"
    print b1.shape
    print "W2"
    print W2.shape
    print "b2"
    print b2.shape
    print "sigmoid_out"
    print sigmoid_out.shape
    print "softmax_output"
    print softmax_output.shape
    print ";;;;;;;;;;;;"
    grad_ce_to_softmax_input = softmax_output - labels;
    print "ce_to_softmax_input"
    print grad_ce_to_softmax_input.shape
    grad_softmax_input_to_b2 = 1;
    grad_ce_to_b2 = grad_ce_to_softmax_input * grad_softmax_input_to_b2
    print "ce_to_b2"
    print grad_ce_to_b2.shape
    grad_ce_to_sigmoid_out = grad_ce_to_softmax_input.dot(W2.T)
    print "ce_to_sigmoid_out"
    print grad_ce_to_sigmoid_out.shape
    grad_ce_to_W2 = np.einsum('ij,ik->ikj', grad_ce_to_softmax_input, sigmoid_out)
    print "ce_to_W2"
    print grad_ce_to_W2.shape
    grad_sigmoid_out_to_sigmoid_in = sigmoid_out * (1 - sigmoid_out)
    grad_ce_to_sigmoid_in = grad_ce_to_sigmoid_out * grad_sigmoid_out_to_sigmoid_in
    print "ce_to_sigmoid_in"
    print grad_ce_to_sigmoid_in.shape
    grad_sigmoid_in_to_b1 = 1
    grad_ce_to_b1 = grad_ce_to_sigmoid_in * grad_sigmoid_in_to_b1
    print "ce_to_b1"
    print grad_ce_to_b1.shape
    grad_ce_to_W1 = np.einsum('ij, ik->ikj', grad_ce_to_sigmoid_in, data)
    print "ce_to_W1"
    print grad_ce_to_W1.shape
    print "<<<<<<<<<<<<"

    gradW1 = grad_ce_to_W1.sum(axis=0)
    gradb1 = grad_ce_to_b1.sum(axis=0)
    gradW2 = grad_ce_to_W2.sum(axis=0)
    gradb2 = grad_ce_to_b2.sum(axis=0)

    ### END YOUR CODE
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    o = np.array([[0.001, 0.001, 0.998], [0.1, 0.1, 0.8]])
    t = np.array([[0.1, 0.8, 0.1], [0, 0, 1]])
    print ce(t, o)
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
