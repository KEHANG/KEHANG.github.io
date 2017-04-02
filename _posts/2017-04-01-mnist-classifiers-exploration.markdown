---
layout: post
title:  "6.036 Project#2: MNIST Classifiers"
date:   2017-04-01 20:01:11 -0500
categories: basic_project
---

This project is about digit classification using the [MNIST database](http://yann.lecun.com/exdb/mnist/). It contains 60,000 training digits and 10,000 testing digits. The goal is to practically explore differenet classifiers and evaluate their performances. The exploration ranges from simplest classifier, e.g.,linear regression with softmax for classification, to nerual networks.

### Classifier 1: Multinomial/Softmax Regression

#### Part 4: report base-line test error

when `temperature parameter = 1`, the test error is 0.1005, implying the linear softmax regression model is able to recognize MNIST digits with around 90%.

#### Part 5: explain temperate parameter effects

Increasing temperature parameter would decrease the probability of a sample $$x^{(i)}$$ being assigned a label that has a large $$\theta$$, and increase for labels with small $$\theta$$. The mathematic explanation is following:

$$  P_j = \frac{exp(\theta_j x / \tau)}{\sum_k exp(\theta_k x / \tau)} $$

$$ \frac{\partial log(P_j)}{\partial \tau} = \frac{1}{\tau^2} \Big[ \frac{\sum_k exp(\theta_k x / \tau) \theta_k x}{\sum_k exp(\theta_k x / \tau)} - \theta_j x \Big]$$

The first term is the bracket is weighted average of $$\theta x$$, so if $$\theta_j x$$ is large, the value of the brackect will be negative, leading to negative $$ \frac{\partial log(P_j)}{\partial \tau} $$.

For small $$\theta_j$$, $$\theta_j x$$ is also small, we have positive $$ \frac{\partial log(P_j)}{\partial \tau} $$.

#### Part 6: temperate parameter effect on test error

During experimentation, we found the test error increases with temperature parameter as follows:

when `temperature parameter = 0.5`, the test error is 0.084

when `temperature parameter = 1.0`, the test error is 0.1005

when `temperature parameter = 2.0`, the test error is 0.1261

Since from Part 5 we know increasing temperature parameter makes probability of large-$$\theta$$ label decrease and that of small-$$\theta$$ label increase, the probability distribution becomes **more uniform** as temperature parameter icreases.

#### Part 7: classify digits by their mod 3 values

Note: in this part we use `temperature parameter = 1.0`.

The error rate of the new labels (mod 3) `testErrorMod3 = 0.0768`.

The error rate of the original labels `testError = 0.1005`.

The classification error decreases because those examples being classified correctly with original labels will continue being classified correctly with new labels, but those being classified wrongly originally will have a chance to be classified correctly with new labels.

#### Part 8: re-train and classify digits by their mod 3 values

After re-training using the new labels (mod 3), `testErrorMod3 = 0.1872`.

#### Part 9: explain test error difference between Part 7 and 8

Compared with `testErrorMod3 = 0.0768` in Part 7, re-training makes the test accuracy worse. It is because now in training step, the algorithm has **less information** (very different digits may have same mod 3 labels) with new labels compared with training using 0-9 digit labels.

### Classification using manually-crafted features

This section we explore one dimensionality reduction approach: PCA and one dimensionality increase approach: polynomial (specifically `cubic` version) feature mapping, to see their effects on test error rates.

#### Part 3: report PCA test error

When we use dimensionality reduction by applying PCA with 18 principal components, the `test error = 0.1483`. This error rate is very similar to the original `d=784` case. It is because PCA ensures these 18 feature values capture the maximal amount of variation in the original 784-denmensional data.

#### Part 4: first 2 principal components visualization

Below is the visualization of the first 2 pricipal components of 100 training data points.

{:.mnist_img}
![Alt text]({{ site.github.url }}/assets/mnist_post/img/plotPC_P2p4.png)

#### Part 5: image reconstruction from PCA-representations

Below are the reconstructions of the first two MNIST images fron their 18-dimensional PCA-representations alongside the originals.

**MNIST 1st Image** (left: reconstructed, right: original)

![Alt text]({{ site.github.url }}/assets/mnist_post/img/image_x_first_img_recon.png){:height="300px" width="360px"} ![Alt text]({{ site.github.url }}/assets/mnist_post/img/image_x_first_img.png){:height="300px" width="360px"}

**MNIST 2nd Image** (left: reconstructed, right: original)

![Alt text]({{ site.github.url }}/assets/mnist_post/img/image_x_second_img_recon.png){:height="300px" width="360px"} ![Alt text]({{ site.github.url }}/assets/mnist_post/img/image_x_second_img.png){:height="300px" width="360px"}

#### Part 6: explicit cubic feature mapping for 2-dimensional case

For $$x = [x_1, x_2]$$ and $$\phi(x)^T \phi(x') = (x^T x' + 1)^3$$, we expand and get

$$\phi(x)^T \phi(x') = x_1^3{x_1'}^3 + x_2^3{x_2'}^3 + 3 x_1^2{x_1'}^2 x_2 {x_2'} + 3 x_1{x_1'} x_2^2 {x_2'}^2$$

$$ + 3 x_1^2{x_1'}^2 + 3 x_2^2{x_2'}^2 + 6 x_1{x_1'} x_2 {x_2'} + 3 x_1 {x_1'} + 3 x_2{x_2'} + 1$$

So the corresponding feature vector is 

$$\phi(x) = (x_1^3, x_2^3, \sqrt{3} x_1^2 x_2, \sqrt{3} x_1 x_2^2, \sqrt{3} x_1^2, \sqrt{3} x_2^2, \sqrt{6} x_1 x_2, \sqrt{3} x_1, \sqrt{3} x_1, 1)^T$$

#### Part 7: re-train using cubic feature mapping

For practical reason, we apply the feature mapping to 10-dimensional PCA representation instead of original 784-dimensional data.

With same settings as base-line case (e.g., temperature parameter = 1), the `test error = 0.0865`.



