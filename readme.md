# Introduction
Why would semi supervised learning ever work? Why would learning representations ever help to do semi supervised learning? In this blog, we seek to work on these questions.

In semisupervised learning, we combine the powers of supervised and unsupervised learning. We care about a prediction task, but we have few labeled samples. Semi supervised learning seeks to use the unlabeled samples to improve the predictive performance for the labeled samples. 

However, using only labeled samples, we can only learn about the distribution of our inputs. At first sight, this does not seem to help the prediction task. More formally, we can learn about p(x) from the unlabeled samples. Yet we care about a predictive distribution p(y|x). Now p(x) does not seem to help anyway in p(y|x). We could learn a generative model p(x|y) and predict using Bayes rule p(y|x) = p(x|y)p(x)/p(y), but generative models are even more complicated to train on small data sets. (Think of learning the distribution over all images from only 500 samples.)

## Representation learning to the rescue! 
Most recent successes in semi supervised learning originated from representation learning. In representation learning, one seeks to learn useful representations for input data. The recent word2vec is a good example, where an object, a word, gets represented by a list of numbers, a representation. Likewise, for image retrieval we learn representations for images. And so on. Now how can a good representation help us for semi supervised learning. 

## Example:
Let's say we have a million images from the data we care about, but we have only 500 labeled samples. Now learning a good representation for these images will help us. Whenever I can represent any image with a small vector of say, 10 numbers. Then I can learn a classification model on this 10 dimensional representations using my 500 labeled samples. Classification on 10 dimensions with 500 samples sounds easy! and it is.

More formally, by learning representions, semisupervised learning factorizes the predictive distribution. Remember that we care about p(y|x). Now with our representations, we learn p(r|x) and our small classification model learns p(y|r). Thus we factorize: p(y|x) = p(y|r)p(r|x). (For the attentive reader, we really do \int_r p(y|r)p(r|x) dr and assume p(r|x) a delta distribution)

# Show me diagrams

Ok, let's do another example with diagrams. We are faced with the following data set: {( , ), ( , )}. The data set has a rather small number of samples. However, based on these samples, what would you think of the decision boundary? Someone queries a point at ( , ), what label would we give it?

// Image of two points, with query
![two_points](https://github.com/RobRomijnders/ssl_rep/blob/master/ssl_rep/im/two_points_query.svg)


// Image of two points with decision boundary, with query
![two_points_boundary](https://github.com/RobRomijnders/ssl_rep/blob/master/ssl_rep/im/two_points_query_boundary.svg)


Given such data set, we would assign the bisector as the decision boundary. And we would label the point to be class XXX

Now comes step two in our example: Someone walks in and gives us a huge addition to our data set. We receive 1000 new data points. However, they carry no label. We have 1000 new unlabeled samples. How can these samples help us? Well, let's plot them:

// Image of two points, with 1000 unlabeled points. 
![two_points_unlabeled](https://github.com/RobRomijnders/ssl_rep/blob/master/ssl_rep/im/two_points_query_unlabeled.svg)


Suddenly, we discover some underlying structure to the data distributioin. Apparently, the data gets sampled from two half moons. One of the half moons carries a sample with label 1. Another half moon carries a sample with label 0. Now we think back to our decision boundary. This time around, what decision boundary would we assign? Also think back to our query point. What label would we assign it using our new knowledge?

// Image of two points, with 1000 unlabeled points, with decision boundary.
![two_points_unlabeled_boundary](https://github.com/RobRomijnders/ssl_rep/blob/master/ssl_rep/im/two_points_query_unlabeled_boundary.svg)


Given these unlabeled samples, we would assign this decision boundary. And we give our query point the label XXX.

I hope this example shows how unlabeled samples can help us in our supervised learning problem. By 'learning' about the underlying data distribution, this will help us to extrapolate from the few labeled samples that we have. 

# Implementation
In this project, we implement exactly the example that we just discussed. We generate such data set ourselves. Manually label the two points at XXX. Then we run the algorithm two times. 

  1. One time we run the algorithm without using the unlabeled data. In that case, we expect the classification accuracy on the test set to be 50%. We have a binary classification problem and the model is not able to learn anything, so the classification accuracy is 1/2 = 50%
  2. The second time we run the algorithm by using semi supervised learning. In that case, we expect the classification accuracy on the test set to be at least higher than 50%. An accuracy higher than 50% indicates that our algorithm has succesfully used the unlabeled data.

To run these experiments, run the `main.py` script twice. You will find a line that says `do_ssl = True`. For experiment 1, set `do_ssl = False`. For experiment 2, set `do_ssl = True`

## Semi supervised algorithm 
For the algorithm implementing semi supervised learning, we choose the [*Mean teacher*](https://arxiv.org/abs/1703.01780) algorithm developed in Harri Valpola's lab. They have an outstanding [implementation](https://github.com/CuriousAI/mean-teacher) on their Github. Please refer to their paper or implementation 

## Results
Running the experiment with `do_ssl = True` yiels an accuracy on the test set of 80%. Running the experiment with `do_ssl = False` yiels an accuracy on the test set of 50%.

## Discussion
The semi supervised algorithm yields an accuracy of 80% on the test set, using only two labeled samples. We remain to ask what prevents the algorithm from reaching 100% accuracy.
A normal algorithm yields 50% accuracy, which is basically a coin flip between the classes


# Further reading

  * [Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](https://arxiv.org/abs/1703.01780)
  	* Implementation in both PyTorch and Tensorflow: [here](https://github.com/CuriousAI/mean-teacher)
  * [Rinu Boney explaining another SSL technique](http://rinuboney.github.io/2016/01/19/ladder-network.html)
  * [Chapelle's book on Semi-Supervised Learning](https://mitpress.mit.edu/books/semi-supervised-learning)
  * [Goodfellow, Courville and Bengio's book on Deep Learning, chapter 15 on representation learning](https://www.deeplearningbook.org/)
