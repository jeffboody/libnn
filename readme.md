Neural Network
==============

Introduction
------------

A neural network is an iterative algorithm that learns to
solve complex problems by performing a gradient descent
optimization to minimize a loss function with respect to
set of training patterns.

Neural networks have be applied to solve a wide range of
problems such as the following.

* Linear Regression
* Non-Linear Prediction
* Classification
* Segmentation
* Noise Removal
* Natural Language Translation
* Interactive Conversation
* Text-to-Image Synthesis

Many different neural network architectures have been
developed which are specialized to handle different problem
types. Some notable examples include the following.

* Fully Connected Neural Networks (FCNN)
* Convolutional Neural Networks (CNN)
* Recurrent Neural Networks (RNN)
* Long Short Term Memory (LSTM)
* Variational Autoencoder Neural Networks (VAE)
* Generative Adversarial Networks (GAN)

The sections below will describe the components, algorithms
and optimization techniques involved in the development of
neural networks.

References

* [CS231n Winter 2016](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC)

Gradient Descent Optimization
-----------------------------

The gradient descent optimization is also known as the
method of steepest descent and can be described using a
mountain analogy. Let's consider how one might navigate
down a mountain with limited visibility due to fog. We can
quickly observe the surrounding area to estimate the slope
of the mountain at our location (e.g. evaluate the gradient
of a differentiable function) and take a step in direction
of steepest descent. However, when the fog is heavy it may
take some time to estimate the slope of the mountain in our
surrounding area. As a result, we may choose to move some
distance in the last known direction of steepest descent
(e.g. momentium). In some cases, this heuristic may also
cause us to move a short distance in the wrong direction.
However, the expectation is that this method is still faster
due to the reduced time spent estimating the mountain slope.
After repeating this process enough times we hope to reach
the base of the mountain.

While this procedure is conceptually simple there are a
number of challanges to implementing the gradient descent
optimization in practice. In particular, we may end up
traversing to a local minima (e.g. a lake), a saddle (e.g. a
flat spot) or a gulch (e.g. zig-zag traversal).
These scenarios can cause the method of steepest descent to
converge slowly or get stuck in a suboptimal local minimum.
Many of the sections below are dedicated to specialized
techniques that are designed to address problems such as
these.

Computation Graph
-----------------

Neural networks may be represented by a computation graph.
A computation graph is an directed acyclic graph (DAG) that
consists of many nodes, each of which implements a function
in the form of Y = f(X,W) that can solve a fragment of the
larger problem. The inputs (X), parameters (W) and outputs
(Y) for the functions may be multi-dimensional arrays known
as tensors. The parameters are trained or learned via the
gradient descent optimization. Additional model parameters
known as hyperparameters are also specified as part of the
neural network architecture (e.g. learning rate).

Nodes are typically organized into layers of similar
functions (e.g. specialized for a particular task) where the
output of one layer is fed into the input of the next layer.
Early neural network architectures were fully connected such
that every output of one layer was connected to every input
of the next layer. In 2012, a major innovation was
introduced by the AlexNet architecture which demonstrated
how to use sparsely connected layers of convolutional nodes
to greatly improve image classification tasks.

As the number of layers (e.g. the neural network depth) and
nodes increases, so does the capacity of a neural network to
solve more complicated problems. However, it's very
difficult to know the amount of capacity required to solve
complex problems. Too little capacity can lead to
underfitting and too much capacity can lead to overfitting.
In general, we wish to have more capacity than is required
solve a problem then rely on regularization techniques to
address the overfitting problem.

The following computation graph shows a simple neural
network with two inputs X = [x1,x2] (e.g. input layer), two
nodes in the first layer [Node11,Node12] (e.g. hidden
layer), two nodes in the second layer [Node21,Node22] (e.g.
output layer) and two outputs Y = [y1,y2]. The neural
network implements Y = f(X,W) in terms of the node functions
f = [f1,f2] and the parameters W = [W11,W12,W21,W22]. Each
parameter variable may represent an array with zero or more
elements.

![Neural Network Example](docs/nn.jpg?raw=true "Neural Network Example")

Forward Pass
------------

A forward pass is performed on the neural network to make a
prediction given some input and simply involves evaluating
functions in the computation graph from the input to the
output.

Backpropagation
---------------

The backpropagation algorithm implements the gradient
descent optimization to learn the function parameters by
minimizing a loss function with respect to the predicted
output (Y) and the desired training output (Yt).

	L(Y,Yt)

The gradient descent opmization states that the function
parameters may be updated to minimize the loss by
subtracting the gradient of the loss with respect to each
function parameter.

	wi -= gamma*dL/dwi

The learning rate (gamma) is a hyperparameter and it was
suggested that a good default is 0.01.

Recall that the loss function is defined in terms of the
predicted output and desired training output so it's not
possible to compute the desired gradient directly. As a
result, we must backpropagate the gradient from loss
function to each function parameter by repeatedly applying
the chain rule. The chain rule allows the desired gradient
to be computed by chaining the gradients of dependent
variables. For example, the chain rule may be applied to the
dependent variables x, y and z as follows.

	dz/dx = (dz/dy)*(dy/dx)

The following gradients may be computed during the forward
pass (i.e. the forward gradients) which will be cached for
use by the backpropagation algorithm.

	dy/dxi = df(X,W)/dxi
	dy/dwi = df(X,W)/dwi

When a node is connected to more than one output node, we
must combine the backpropagated loss gradient.

	dL/dy = SUM(dLi/dy)

The update gradient may now be determined using the loss
gradient, the forward gradients and the chain rule.

	dL/dwi = (dL/dy)*(dy/dwi)

The backpropagated gradient may also be determined using the
loss gradient, the forward gradients and the chain rule.

	dL/dxi = (dL/dy)*(dy/dxi)

In summary, the backpropagation algorithm may be applied by
repeating the following steps for each training pattern.

* Forward Pass
* Forward Gradients
* Compute Loss
* Combine Loss
* Update Parameters
* Backpropagate Loss

The following computation graph shows the backpropagation
algorithm using our example from earlier.

![Neural Network Backpropagation](docs/nn-backprop.jpg?raw=true "Neural Network Backpropagation")

References

* [CS231n Winter 2016: Lecture 4: Backpropagation, Neural Networks 1](https://www.youtube.com/watch?v=i94OvYb6noo)

Loss Function
-------------

The loss function is a hyperparameter and the choice of loss
function depends upon the type of problem that the neural
network is solving. The two main types of problems are
regression and classification. Regression problems consist
of predicting a real value quantity while classification
problems consist of classifying a pattern in terms of one or
more classes.

The Mean Squared Error (MSE) and Mean Absolute Error (MAE)
are the most commonly used loss functions for regression
problems. The MSE is typically used unless the training data
has a large number of outliers. This is because the MSE is
highly sensitive to outliers due to the squared term.

The Binary Cross-Entropy (BCE) may also be used for
regression problems when the output values are in the
range [0, 1]. Note, however, that the BCE plot does not
converge to 0.0 loss when yi == yti. This fact seems to
contradict the claim that BCE may be used for regression
problems.

	MSE
	L(Y,Yt) = (1/n)*SUM((yi - yti)^2)
	dL/dyi  = 2*(yi - yti)

	MAE
	L(Y,Yt) = (1/n)*SUM(|yi - yti|)
	dL/dyi  = (yi - yti)/|yi - yti|

	BCE
	L(Y,Yt) = (-1/n)*SUM(yti*log10(yi) + (1 - yti)*log10(1 - yi))
	dL/dyi  = -(yi - yti)/(ln(10)*(yi - 1)*yi)

Add a small epsilon to avoid divide-by-zero problems.

The following plots show the loss for yi and yti.

![Loss Functions](docs/loss/loss.jpg?raw=true "Loss Functions")

The Categorical Cross Entropy Loss is the most commonly used
loss function for classification problems. Additionally, the
Variational Autoencoder Loss is often used for autoencoder
neural networks.

References

* [Loss Functions and Their Use In Neural Networks](https://towardsdatascience.com/loss-functions-and-their-use-in-neural-networks-a470e703f1e9)
* [Binary Crossentropy in its core!](https://medium.com/analytics-vidhya/binary-crossentropy-in-its-core-35bcecf27a8a)
* [Introduction to Autoencoders? What are Autoencoders Applications and Types?](https://www.mygreatlearning.com/blog/autoencoder/)
* [Tensorflow Cross Entropy for Regression?](https://stats.stackexchange.com/questions/223256/tensorflow-cross-entropy-for-regression)
* [On denoising autoencoders trained to minimise binary cross-entropy](https://arxiv.org/pdf/1708.08487.pdf)
* [Variational autoencoders](https://www.jeremyjordan.me/variational-autoencoders/)
* [Derivative Calculator](https://www.derivative-calculator.net/)

Perceptron
----------

The perceptron is the main type of node which is used by
neural networks and implements a function that roughly
mimics biological neurons. This function consists of a
weighted sum of inputs plus a bias term followed by an
activation function.

	W      = [[w1,w2,...,wn],b]
	f(X,W) = fact(SUM(xi*wi) + b)

The weights (w) and the bias (b) are the parameters that
are learned by the neural network while the activation
function is a hyperparameter.

The following computation graph shows the perceptron which
can be visualized as a compound node consisting of a
weighted sum and a separate activation function.

![Perceptron](docs/nn-perceptron.jpg?raw=true "Perceptron")

The following computation graph shows the backpropagation
algorithm for the perceptron node. Note that the activation
function does not include any function parameters so the
update step may be skipped. The perceptron node
implementation may also choose to combine the compound node
into a single node by simply substituting equations.

![Perceptron Backpropagation](docs/nn-perceptron-backprop.jpg?raw=true "Perceptron Backpropagation")

To gain a better understanding of how the perceptron works
it's useful to compare the perceptron function with the
equation of a line. The perceptron weights are analogous to
the slope of the line while the bias is analogous to the
y-intercept.

	y = m*x + b

The weighted average is also analogous to the dot product
operation between the two vectors which is maximized when
the vectors point in the same direction.

	y = W.X + b = |W|*|X|*cos(theta) + b

From the biological perspective, a neuron may activate at
different strength depending on if some threshold was
exceeded. The activation function is generally designed to
mimic this behavior, however, in practice the designer may
choose any function desired to achieve a particular effect.
For example, an activation function may be selected to model
a non-linear operation, to threshold outputs or to predict a
probability.

References

* [A Brief Introduction to Neural Networks](https://www.dkriesel.com/en/science/neural_networks)

Activation Functions
--------------------

The following activation functions and their derivatives
may be used depending on the situation. The hidden layers
typically use one activation function and while the output
layer may use a different activation function. For the
hidden layers it's recommended to use either ReLU or Tanh.
For the output layer it's recommended to use Linear, Tanh,
Sigmoid or Softmax (classification). The activation function
which should be selected for the output layer may depend on
the desired range of your output. For example, probability
outputs exist in the range 0.0 to 1.0 which makes the
logistic function a good choice.

The vanishing gradient problem may occur for the tanh and
logistic functions because the gradient approaches zero
when the input is large. This problem is more likely to
occur in the earlier layers of a deep neural network. If
this occurs it is recommended to use the ReLU function
instead which does not cause small gradients.

ReLU (Rectified Linear Unit)

	f(x)  = max(0, x)
	df/dx = 0 for x < 0
	      = 1 for x >= 0
	range = [0,infinity]

PReLU (Parametric Rectified Linear Unit or Leaky ReLU)

	f(x)  = max(a*x, x)
	df/dx = a for x < 0
	      = 1 for x >= 0
	range = [-infinity,infinity]

	a is typically 0.01

Tanh

	f(x)  = tanh(x) = 2/(1 + exp(-2*x)) - 1
	df/dx = 1 - f(x)
	range = [-1,1]

Logistic

	f(x)  = 1/(1 + exp(-x))
	df/dx = f(x)*(1 - f(x))
	range = [0,1]

Linear (Identity)

	f(x)  = x
	df/dx = 1
	range = [-infinity,infinity]

References

* [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
* [How to Choose an Activation Function for Deep Learning](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)
* [The Vanishing Gradient Problem](https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484)

Parameter Initialization
------------------------

The perceptron weights must be initalized correctly to
ensure that the gradient descent optimization works
properly. Incorrect weight initialization can lead to
several problems including the symmetry problem (e.g.
weights initialized to zero resulting in symmetric partial
derivatives), slow learning and divergence (e.g. the output
grows to infinity).

The following perceptron weight initialization methods are
recommended depending on the desired activation function.

Xavier Method

	fact = tanh or logistic
	m    = number of inputs
	min  = -1/sqrt(m)
	max  = 1/sqrt(m)
	w    = randUniformDistribution(min, max)

Normalized Xavier Method

	fact = tanh or logistic
	m    = number of inputs
	n    = number of outputs
	min  = -sqrt(6)/sqrt(m + n)
	max  = sqrt(6)/sqrt(m + n)
	w    = randUniformDistribution(min, max)

He Method

	fact  = ReLU or PReLU
	m     = number of inputs
	mu    = 0.0
	sigma = sqrt(2/m)
	w     = randNormalDistribution(mu, sigma)

The perceptron bias on the other hand are typically
initialized to zero as they are not impacted by the symmetry
problem.

Other parameter types may exist within the neural network
however each may have its own unique initialization
requirements.

References

* [Weight Initialization for Deep Learning Neural Networks](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/)
* [Initializing neural networks](https://www.deeplearning.ai/ai-notes/initialization/index.html)
* [Bias Initialization in a Neural Network](https://medium.com/@glenmeyerowitz/bias-initialization-in-a-neural-network-2e5d26fed0f0)
* [3 Common Problems with Neural Network Initialization](https://towardsdatascience.com/3-common-problems-with-neural-network-initialisation-5e6cacfcd8e6)

Data Centering and Scaling
--------------------------

Data centering and scaling should be performed on the input
layer on a per-channel (i) basis to normalize the data to
zero mean and unit variance. When the input layer contains
images it's common to perform the zero mean but skip the
unit variance. It may also be beneficial to perform data
centering and scaling on a per-image basis rather than
per-channel (e.g. face recognition). Data whitening may also
be applied by performing PCA and transforming the covariance
matrix to the identity matrix.

	Yi = (Xi - Mean(Xi))/StdDev(Xi)

Add a small epsilon to avoid divide-by-zero problems.

This transformation improves the learning/convergence rate by
avoiding the well known zig-zag pattern where the gradient
descent trajectory oscilates back and forth along one
dimension.

References

* [Batch Norm Explained Visually - How it works, and why neural networks need it](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739)
* [CS231n Winter 2016: Lecture 5: Neural Networks Part 2](https://www.youtube.com/watch?v=gYpoJMlgyXA&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&index=5)

Batch Size
----------

As described earlier, the backpropagation algorithm may be
applied by repeating the following steps for each training
pattern.

* Forward Pass
* Forward Gradients
* Compute Loss
* Combine Loss
* Update Parameters
* Backpropagate Loss

The batch size refers to the number of training patterns
that are processed in the forward pass before the loss is
backpropagated.

The algorithm variations on batch size are as follows.

* Stochastic Gradient Descent (SGD)
* Mini-batch Gradient Descent
* Gradient Descent

The SGD method performs training by processing one patttern
at a time. The mini-batch method performs training by
subdividing the training set into batches and processes one
batch at a time. And finally, the gradient descent method
processes the entire training set at once.

The computation of gradients for the mini-batch and gradient
descent methods must be adjusted to compute the average
gradient across the batch. However, it's unclear at what
point the average gradient should be computed. For example,
the following forward/backprop gradients can be averaged
before or after backpropagation. Some references suggest
that averaging should happen before while the batch
normalization algorithm suggests that averaging should
happen after.

	dY/dX = { 4, 7 }
	dL/dY = { 3, 6 }
	dL/dX = (3 + 6)/2 * (4 + 7)/2 = 24.75 (average before)
	dL/dX = (3*4 + 6*7)/2         = 27.0  (average after)

The advantages of each approach includes the following.

Stochastic Gradient Descent

* Simplest to implement
* Less memory is required for cached state

Mini-Batch Gradient Descent

* Mini-batch is typically the preferred method
* Smoother gradients results in more stable convergence
* Backpropagation is amortized across the mini-batch
* Implementations may vectorize code across mini-batches
* Improve convergence using Batch Normalization

Gradient Descent

* Converges slowly
* Computationally expensive
* Impractical for real data sets

The batch size is a hyperparameter and it was suggested that
a good default is 32. It may also be possible to increase
the batch size over time as this reduces the variance in the
gradients when approaching a minimal solution.

Note that a training set may also be processed multiple
times and each pass over the training set is referred to as
an epoch.

References

* [A Gentle Introduction to Mini-Batch Gradient Descent and How to Configure Batch Size](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)
* [Variable batch-size in mini-batch gradient descent](https://www.reddit.com/r/MachineLearning/comments/481f2v/variable_batchsize_in_minibatch_gradient_descent/)
* [Sum or average of gradients in (mini) batch gradient descent?](https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent)

Batch Normalization
-------------------

Batch Normalization (BN) is a neural network layer that is
closely related to data centering and scaling. The main
difference is that BN includes a pair of learnable
parameters per-channel/per-filter to scale (gamma) and
offset (beta) data samples. It is important to note that
the function is differentiable as is required for
backpropagation.

The following computation graph shows the batch
normalization algorithm.

![BN Overview](docs/nn-batch-norm-overview.jpg?raw=true "BN Overview")

The backpropagation equations derived by the original BN
paper are reportedly 3x slower than these derived by Kevin
Zakka's blog post.

![BN Backpropagation](docs/nn-batch-norm-backprop.jpg?raw=true "BN Backpropagation")

As per the original algorithm, the per-channel/per-filter
mean and variance are also calculated during training from
the mini-batch. Running averages of these values are also
calculated during the training which are subsequently used
when making predictions. The exponential average momentum is
a hyperparameter and it was suggested that a good default is
0.99.

	Xmean_ra = Xmean_ra*momentum + Xmean_mb*(1 - momentum)
	Xvar_ra  = Xvar_ra*momentum + Xvar_mb*(1 - momentum)

The instance normalization technique is an alternative
approach to computing running averages and simply involves
using the statistics of the test batch rather than the
aggregated statistics of the training batch. The instance
normalization technique was referenced by the Pix-To-Pix GAN
paper.

There is some discussion as to the best place for the BN
layer. The original paper placed this layer between the
perceptron weighted average and the activation function. It
was suggested to place the BN layer before the ReLU/PReLU
activation functions but after tanh/logistic activation
functions. When placed per the original paper, the
perceptron bias is redundant with the beta offset.

It was also mentioned that BN can be performed on the input
layer in place of data centering and scaling. However, it's
unclear if the the mean and standard deviation should be
used from the training set or mini-batch in this case.

The neural network may learn the identity operation (e.g.
beta is the mean and gamma is the inverse of standard
deviation) should this be optimal.

Note that the tensor operations are component-wise.

Be sure to add a small epsilon to avoid divide-by-zero
problems.

Weight Normalization and Layer Normalization are additional
related techniques however these won't be covered at this
time since it's unclear when or if these techniques are
better.

References

* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
* [Batch Norm Explained Visually - How it works, and why neural networks need it](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739)
* [Batch normalization: What it is and how to use it](https://www.youtube.com/watch?v=yXOMHOpbon8)
* [CS231n Winter 2016: Lecture 5: Neural Networks Part 2](https://www.youtube.com/watch?v=gYpoJMlgyXA&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&index=5)
* [L2 Regularization versus Batch and Weight Normalization](https://arxiv.org/pdf/1706.05350.pdf)
* [Moving average in Batch Normalization](https://jiafulow.github.io/blog/2021/01/29/moving-average-in-batch-normalization/)
* [Deriving the Gradient for the Backward Pass of Batch Normalization](https://kevinzakka.github.io/2016/09/14/batch_normalization/)
* [Understanding the backward pass through Batch Normalization Layer](http://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)
* [Matrix form of backpropagation with batch normalization](https://stats.stackexchange.com/questions/328242/matrix-form-of-backpropagation-with-batch-normalization)

Learning Rate
-------------

The learning rate is arguably the most important
hyperparameter that affects how fast the gradient descent
converges. Large learning rates can cause the gradient
descent to become unstable leading to a failure to converge.
On the other hand, small learning rates may cause the
gradient descent to converge slowly or get trapped in local
minima.

Variable learning rates may be achieved by using a high
learning rate in the beginning while slowly decreasing the
learning rate after each epoch. The learning rate typically
starts in the range of 0.01 and 0.001. The following
adaptive techniques have also been proposed to adjust the
effective learning rate for faster convergence.

* Momentum
* RMSProp
* Adam, AdamW and ND-Adam
* Cyclical Learning Rates

The learning rate is also impacted by L2 regularization when
combined with Batch Normalization. The L2 regularization
technique may also be used in conjunction with the adaptive
techniques listed above (except AdamW).

References

* [CS231n Winter 2016: Lecture 6: Neural Networks Part 3 / Intro to ConvNets](https://www.youtube.com/watch?v=hd_KFJ5ktUc&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC)
* [A Visual Explanation of Gradient Descent Methods (Momentum, AdaGrad, RMSProp, Adam)](https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c)

Momentum Update
---------------

The momentum update is a per parameter adaptive technique
that changes the update by including an exponentially
decaying velocity term. When successive updates are
performed in the same direction, the velocity term picks up
speed and increases the effective learning rate.
Alternatively, when successive updates are performed in
opposite directions (e.g. oscilation) the velocity is
dampened (e.g. friction). The increased velocity can cause
the update to escape a local minima and power through
plateaus. On the other hand, it may overshoot a desired
minimum and will need to backtrack to reach the minimum.

Recall that the backpropagation update is the following.

	wi = wi - gamma*dL/dwi

The momentum update is the following.

	vi = mu*vi - gamma*dL/dwi
	wi = wi + vi

The decay rate (mu) is a hyperparameter and it was suggested
that good defaults are 0.5, 0.9 or 0.99. The decay rate may
be varied across epochs beginning from a value of 0.5 while
slowly increasing towards 0.99.

The Nesterov momentum update is an improved update which
uses the "lookahead" gradient for faster convergence rates
as follows.

	v0i = v1i
	v1i = mu*v0i - gamma*dL/dwi
	wi  = wi - mu*v0i + (1 + mu)*v1i

References

* [CS231n Winter 2016: Lecture 6: Neural Networks Part 3 / Intro to ConvNets](https://www.youtube.com/watch?v=hd_KFJ5ktUc&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC)
* [A Visual Explanation of Gradient Descent Methods (Momentum, AdaGrad, RMSProp, Adam)](https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c)

RMSProp Update
--------------

The RMSProp (Root Mean Square Propagation) update is another
per parameter adaptive technique that scales the update step
size by the square root of an exponentially decaying
gradient scaled term. When successive updates are performed
using large gradients, the update step size is scaled down.
Alternatively, when successive updates are performed using
small gradients, the update step size is scaled up. As a
result, the scaling term reduces the zig-zag pattern by
normalizing the update step size such that we can proceed
directly towards the local minimum.

Recall that the backpropagation update is the following.

	wi = wi - gamma*dL/dwi

The RMSProp update is the following.

	g2i = nu*g2i + (1 - nu)*(dL/dwi)^2
	wi  = wi - gamma*(dL/dwi)/sqrt(g2i)

The decay rate (nu) is a hyperparameter and it was
suggested that a good default is 0.99.

Add a small epsilon to avoid divide-by-zero problems.

References

* [CS231n Winter 2016: Lecture 6: Neural Networks Part 3 / Intro to ConvNets](https://www.youtube.com/watch?v=hd_KFJ5ktUc&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC)
* [A Visual Explanation of Gradient Descent Methods (Momentum, AdaGrad, RMSProp, Adam)](https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c)

Adam, AdamW and ND-Adam Update
------------------------------

The Adam update is another per-parameter adaptive technique
known as "Adaptive Moment Estimation" which combines
features from the momentum and RMSProp update techniques.
The momentum component increases the overall speed while
the RMSProp component normalizes the gradient scale in
different directions. The design of the original Adam
algorithm includes a L2 regularizaion term, however, it was
discovered that the term was placed incorrectly. As a
result, the original Adam algorithm failed to generalize
well leading to suboptimal solutions. AdamW and ND-Adam were
proposed to address the shortcomings of Adam, however, it's
unclear these improvements can supplant SGD+Momentum+L2
Regularization in practice.

References

* [ADAM: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980.pdf)
* [Why AdamW matters](https://towardsdatascience.com/why-adamw-matters-736223f31b5d)
* [Fixing Weight Decay Regularization in Adam](https://arxiv.org/pdf/1711.05101v2.pdf)
* [Decoupled Weight Decay Regularization](https://arxiv.org/pdf/1711.05101.pdf)
* [Normalized Direction-Preserving Adam](https://arxiv.org/pdf/1709.04546.pdf)

Cyclical Learning Rate
----------------------

The cyclical learning rate algorithm claims that it's best
to use a triangle learning rate policy where the learning
rate varies between a bounds. A key aspect of their claims
is that a high learning rate might have a short term
negative effect yet lead to a longer term beneficial effect.
Their policy helps to eliminate the guesswork in selecting
the learning rate manually, is easy to implement and does
not require additional computation expense.

References

* [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186.pdf)

L1/L2 Regularization
--------------------

L1/L2 regularization is a technique that was originally
designed to reduce data overfitting by adding a term to the
loss function which penalizes the variance of the function
parameters. However, it's important to note that L2
regularization (presumably L1 as well) has no regularization
effect when combined with normalization techniques such as
Batch Normalization. However, if L2 regularization is not
used, then the norm of the weights tends to increase over
time. As a result, the effective learning rate decreases.
While this may be a desirable property, it can be difficult
to control and may interfere with explicit attempts to
control the backpropagation learning rate. As such, it seems
best to combine L2 regularization with a normalization
technique.

Loss Function with L1 Regularization

	LR(lambda,W,Y,Yt) = L(Y,Yt) + lambda*SUM(|wi|)
	dLR/dwi = dL/dwi + lambda*wi/|wi|

Loss Function with L2 Regularization

	LR(lambda,W,Y,Yt) = L(Y,Yt) + lambda*SUM(wi^2)
	dLR/dwi = dL/dwi + 2*lambda*wi

Keep in mind that the regularization term affects the
backpropagation algorithm by adding an additional term to
the update parameter step since dL/dwi is replaced by
dLR/dwi. Some results also suggest that the regularization
term is not required for the perceptron bias parameter since
it does not seem to impact the final result. The lambda
term is a regularization hyperparameter and it was suggested
that a good default is 0.01.

To explain how regularization works in the absense of
normalization, let's consider how the L2 regularization term
affects the following example.

	X  = [1,1,1,1]
	W1 = [1,0,0,0]
	W2 = [0.25,0.25,0.25,0.25]

The perceptron output is the same for each parameter vector
however the regularization term prefers W2.

	SUM(xi*w1i) == SUM(xi*w2i) = 1.0
	SUM(w1i^2)  = 1.0
	SUM(w2i^2)  = 0.25

The W2 parameters are prefered since they are more
generalized across inputs and therefore reduce the variance
caused by a single outlier.

References

* [CS231n Winter 2016: Lecture 3: Linear Classification 2, Optimization](https://www.youtube.com/watch?v=qlLChbHhbg4&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&index=3)
* [Regularization in a Neural Network | Dealing with overfitting](https://www.youtube.com/watch?v=EehRcPo1M-Q)
* [L2 Regularization versus Batch and Weight Normalization](https://arxiv.org/pdf/1706.05350.pdf)
* [Chapter 8 Training Neural Networks Part 2](https://srdas.github.io/DLBook/ImprovingModelGeneralization.html)

Skip Connections
----------------

A skip connection is an alternate path through the neural
network where the output of one layer may be fed forward to
skip one or more layers. The skip connection is typically
joined back to the neural network through addition or
concatenation of a subsequent layer with compatible
dimensions. The (w,h,d) dimensions must match for addition
and the (w,h) dimensions must match for concatenation. Some
example neural networks that use skip connections are
ResNet (residual network) using addition and DenseNet using
concatenation.

The advantages of skip connections are.

* Improved gradient flow to reduce vanishing gradients
* Enables feature reusability
* Recover spatial information lost during downsampling
* Stabilize training and convergence

See the Autoencoder section for more details.

References

* [Intuitive Explanation of Skip Connections in Deep Learning](https://theaisummer.com/skip-connections/)

Gradient Clipping
-----------------

Gradient clipping is a technique that can help to address
numerical overflow/underflow problems for gradients.

	gcw = 1.0
	if(norm(g) > c)
		gcw = c/norm(g)

Recall that the backpropagation update is the following.

	wi = wi - gamma*dL/dwi

The clipped gradient would be the following.

	wi = wi - gamma*gcw*dL/dwi

It was suggested that a good default for c is 1.0 or 10.0.

References

* [Gradient Clipping](https://paperswithcode.com/method/gradient-clipping)
* [How to Avoid Exploding Gradients With Gradient Clipping](https://machinelearningmastery.com/how-to-avoid-exploding-gradients-in-neural-networks-with-gradient-clipping/)
* [Accurate Image Super-Resolution Using Very Deep Convolutional Networks](https://arxiv.org/pdf/1511.04587v2.pdf)

Dropout
-------

Dropout is a regularization technique that may be applied to
a neural network layer where a subset of nodes may be
randomly disabled. Regularization is achieved through the
reduction in capacity which forces the neural network to
increase generalization. The dropout algorithm may also be
viewed as selecting a random layer from a large ensemble of
layers that share parameters.

Nodes which have been dropped will not contribute to the
loss during the forward pass and therefore will also not
participate in backpropagation. The dropout probability is a
hyperparameter and it was suggested that a good default is
0.5. However, as a general rule, layers with many parameters
may benefit more from a higher dropout probability. At least
one node must be active.

During training, the output of the layer is attenuated due
to the dropped out nodes. However, during prediction, the
entire set of nodes are used. This results in a change of
scale for the layer output which causes problems for
subsequent layers. The scale can be adjusted during training
by applying the Inverted Dropout algorithm where output
nodes are scaled as follows.

	scale = total/active

References

* [CS231n Winter 2016: Lecture 6: Neural Networks Part 3 / Intro to ConvNets](https://www.youtube.com/watch?v=hd_KFJ5ktUc&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&index=6)

Data Augmentation
-----------------

The size of a training set may be artifically increased
through the use of various data augmentation techniques to
help reduce overfitting and increase generalization.

For example, facial recognition may benefit by flipping
faces horizontally when they are lit from different
directions.

References

* [A Complete Guide to Data Augmentation](https://www.datacamp.com/tutorial/complete-guide-data-augmentation)

Hyperparameter Tuning
---------------------

There are many types of hyperparameters that must be
determined when designing the neural network such as the
following.

* Learning Rate
* Exponential Decay Rates
* Regularization Rates
* Loss Function
* Activation Functions
* Batch Size
* Layer Types
* Network Capacity (e.g. Node/Filter/Layer Count)

The selection of hyperparameters is a challanging problem
because an exaustive search of the hyperparameter space
leads to a combinatorial explosion. The search algorithm may
also be limited in terms of the processing resources
required to validate each selection of hyperparameters. As
such, it is not possible to exahaustively validate all
choices of hyperparameters. One commonly used approach to
tackle this problem is to perform a random search across the
range of potential hyperparameter values to see which ones
provide the lowest loss. Additional experiments may be
performed to zero in on the best hyperparameters by making
educated guesses as to which parameters need further
optimization.

The cross-validation technique may also be applied to
determine how well the model generalizes across different
training sets. In this case, the training set is divided
into folds (e.g. a 5-fold training set) and the training
algorithm is performed on each fold. The mean and variance
of the loss may be analyzed (see Yerrorlines in gnuplot) to
determine the optimal hyperparameter values and to determine
how well the model generalizes.

More advanced techniques that should be considered in the
future include Bayesian Optimization, Tree Parzen
Estimators (TPEs), Covariance Matrix Adaptation Evolution
Strategy (CMA-ES) and population based training techniques
(e.g. genetic algorithms).

References

* [CS231n Winter 2016: Lecture 2: Data-driven approach, kNN, Linear Classification 1](https://www.youtube.com/watch?v=8inugqHkfvE&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC)
* [Algorithms for Hyper-Parameter Optimization](https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf)
* [Hyper-Parameter Optimization: A Review of Algorithms and Applications](https://arxiv.org/pdf/2003.05689.pdf)
* [CMA-ES for Hyperparameter Optimization of Deep Neural Networks](https://arxiv.org/pdf/1604.07269.pdf)
* [gnuplot 5.4](http://www.gnuplot.info/docs_5.4/Gnuplot_5_4.pdf)

Convolutional Neural Networks (CNN)
-----------------------------------

The following computation graphs shows the derivation of the
backpropagation algorithm for the convolution layer.

![Convolution (Valid Padding)](docs/nn-conv-valid.jpg?raw=true "Convolution (Valid Padding)")

References

* [What is Transposed Convolutional Layer?](https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11)

Autoencoders
------------

Autoencoder neural networks are designed to output the
same values as their input and typically make use of a
bottleneck in the hidden layers. The bottleneck layer is
useful to prevent the neural network from simply passing the
input values directly to the output. This form of neural
network has many potential uses such as noise removal,
upscaling, and compression.

The NN library includes a coder layer which is designed to
help implement autoencoder neural networks more easily while
adhering to the restrictions described below.

Autoencoder neural networks often times make use of skip
connections and residual neural networks. However, the order
of layers in a residual neural network is important for the
network to function correctly. As described in the Identity
Mappings paper the skip connections for residual networks
should be placed after the convolution layer but before the
Batch Normalization layer.

* Convolution
* Skip Fork
* Batch Normalization
* Activation Function
* Convolution
* Batch Normalization
* Activation Function
* Convolution
* Skip Add

Some autoencoder neural networks avoid using batch
normalization altogether due to the increased computation,
memory requirements and increased loss (due to incorrect
layer order). Batch normalization is also known to provide a
greater benefit to earlier layers in the neural network. To
facilitate these designs the coder layer allows for multiple
convolution layers to follow a single batch normalization
layer.

As described in the Batch Normalization section, the bias of
the convolutional layer is also disabled when followed by a
batch normalization layer.

The bottleneck layer is typically implemented by
a series of encoder layers followed by a series of decoder
layers. The encoder layers include a pooling or downscaling
operation while the decoder layers include an upscaling
operation. This operation may also be applied automatically
by the coder layer.

See the Skip Connections section for more details.

References

* [Normalization is dead, long live normalization!](https://iclr-blog-track.github.io/2022/03/25/unnormalized-resnets/)
* [It Is Necessary to Combine Batch Normalization and Skip Connections](https://towardsdatascience.com/its-necessary-to-combine-batch-norm-and-skip-connections-e92210ca04da)
* [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf)
* [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
* [Machine Learning Super-Resolution](https://artoriuz.github.io/blog/super_resolution.html)
* [Accurate Image Super-Resolution Using Very Deep Convolutional Networks](https://arxiv.org/pdf/1511.04587v2.pdf)
* [Super-Resolution of Multispectral Satellite Images Using Convolutional Neural Networks](https://arxiv.org/pdf/2002.00580.pdf)
* [Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections](https://arxiv.org/pdf/1603.09056.pdf)

Generative Adversarial Networks (GAN)
-------------------------------------

Generative Adversarial Networks (GANs) loosely refers to a
neural network architecture that pits two or more networks
against one another in a game to maximize their own value.
GANs typically include a discriminator network D and a
generator network G. The role of the discriminator is to
classify it's input as belonging to the distribution of real
samples or to the distribution of generated samples (aka
fake samples). An alternative interpretation of the role for
the discriminator (as suggested by the Pix-To-Pix GAN) is
that it acts as a trainable loss function. The role of the
generator is to produce samples that the discriminator is
unable to differentiate with real samples. The following
diagram shows the classical GAN network structure.

![Generative Adversarial Network](docs/gan-network.jpg?raw=true "Generative Adversarial Network")

The descriminator network typically outputs a single value
that the objective function uses to determine if the input
was real or generated. Alternatively, the Patch GAN
described by Pix-To-Pix GANs uses multiple outputs. The
purpose of multiple outputs in this case is to reduce the
number of network parameters and reduce tiling artifacts in
the generator network output. A 70x70 Patch GAN refers to
the size of the receptive field of the discriminator output.

The generator network only receives random inputs z and
therefore must learn to map the uniform distribution pz(x)
onto the probability distribution of the real data pdata(x)
in order to generate samples that look like real data. The
z input is also known as the latient space. It has been
shown that walking the latient space (e.g. through vector
addition or interpolation) can cause the generator to
produce variations on the outputs associated with the
latient vectors.

The GAN objective function is closely related to the binary
cross entropy loss function and is used to keep score
between the discriminator and the generator.

	minG maxD V(D,G) = Ex~pdata(x)[log(D(x)] + Ez~pz(z)[log(1 - D(G(z)))]

An objective function may also be referred to as a loss,
cost or error function when we seek to minimize the value as
is typical for neural networks. The expected value function
(e.g. E[]) simply returns the mean value similar to the mean
squared error (MSE) loss function. The notation x\~pdata(x)
simply states that x is an input sample to D from the real
data with a probability density function pdata(x). The
notation z\~pz(z) simply states that z is a random input
sample to G from a uniform distribution function pz(z). The
first term of the GAN objective function is used with real
samples and the second term is used with generated samples.

The GAN training procedure described by the original GAN
paper consists of a two step process that is repeated for
each iteration.

Update the descriminator.

1. Select a minibatch of m samples of z from pz(z)
2. Sample a minibatch of m samples of x from pdata(x)
3. Update the descriminator by ascending its stochastic gradient
4. dV/dD[1/m SUM(1, m, log(D(xi)) + log(1 - D(G(zi))))]

Update the generator.

1. Select a minibatch of m samples of z from pz(z)
2. Update the generator by descending its stochastic gradient
3. dV/dG[1/m SUM(1, m, log(1 - D(G(zi))))]

In practice, it was recommended to maximize log(D(G(z)))
during training since the GAN objective function above may
not provide sufficient gradient for G to learn well.

The process of training the generator involves a few subtle
details. The generator receives an input z which consists of
M random samples as input. These samples may be fed into a
multi-layer perceptron to produce N outputs. These N outputs
are reshaped into a tensor that is typically fed into a CNN.
Secondly, the generator network not only feeds it's output
Y=G(z) into the discriminator network but also receives the
backprop gradient dL/dY from the discriminator network
input. This differs from a non-GAN neural network where the
backprop gradient is discarded at the input. Finally, the
parameter update of the descriminator should be disabled
when training the generator. This is required due to the
fact that the descriminator update is performed by ascending
the stochastic gradient and the generator update is
performed by descending the stochastic gradient.

Notable examples of classic GANs described above include
Deep Convolutional GANs (DCGANs) and Progressive GANs.
The DCGAN paper is generally credited with proposing one of
the first viable GAN architectures which was achieved by
incorporating a number of previously known techniques such
as batch normalization, strided convolutions, ReLU
activation functions and the Adam optimizer. The main
contribution of the Progressive GAN paper is a training
framework which starts at low resolution while progressively
blending upscaling layers to produce a final high resolution
output. In addition, they propose a technique to increase
variation and propose alternative techniques for
normalization (e.g. non-batch normalization). Progressive
GANs also use nearest neighbor filtering for upscaling,
average pooling for downscaling, 1x1 convolution to project
tensors to/from RGB, the Adam optimizer, the leaky ReLU and
the Wasserstein GAN objective function.

Many of the GAN variations which have been developed fall
under the category of Conditional GANs. Conditional GANs may
combine/replace the z input with a conditional class label
(e.g. using one hot encoding), conditional image or other
conditional information to guide the generator in producing
the desired output. Notable examples of Conditional GANs
include Pixel-To-Pixel GAN (paired image-to-image
translation) and Cycle GAN (unpaired image-to-image
translation). The Pixel-To-Pixel GAN paper proposes a
generic architecture that incorporates U-Net skip
connections for the generator, a custom cGAN + L1 objective
fuction combined with a Patch GAN, the Adam optimizer, batch
normalization with instance normalization, dropout in the
generator (to increase variation), strided convolutions, and
ReLU/leaky ReLU activation functions.

The GAN objective function may also be selected by the
network architecture in the same way that the loss function
is chosen by CNN networks. Some examples of additional
objective functions include Least Squares GANs, Wasserstein
GAN, feature matching (Improved Techniques for Training
GANs), cGAN + L1 (Pixel-To-Pixel GAN) and the Cycle GAN. The
advantages of these objective functions include improved
stability and reduced probability of mode collapse. The
Cycle GAN enforces transitivity in order to perform
image-to-image translation with unpaired image collections.

References

* [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)
* [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/pdf/1701.00160.pdf)
* [Introduction to GANs, NIPS 2016](https://www.youtube.com/watch?v=9JpdAg6uMXs)
* [Classic - Generative Adversarial Networks - Paper Explained](https://www.youtube.com/watch?v=eyxmSmjmNS0)
* [Must-Read Papers on GANs](https://towardsdatascience.com/must-read-papers-on-gans-b665bbae3317)
* [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf)
* [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)
* [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)
* [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)
* [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf)
* [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)
* [Least Squares Generative Adversarial Networks](https://arxiv.org/pdf/1611.04076.pdf)
* [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)
* [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf)

License
=======

The NN library was implemented by
[Jeff Boody](mailto:jeffboody@gmail.com)
under The MIT License.

	Copyright (c) 2023 Jeff Boody

	Permission is hereby granted, free of charge, to any person obtaining a
	copy of this software and associated documentation files (the "Software"),
	to deal in the Software without restriction, including without limitation
	the rights to use, copy, modify, merge, publish, distribute, sublicense,
	and/or sell copies of the Software, and to permit persons to whom the
	Software is furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included
	in all copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
	THE SOFTWARE.
