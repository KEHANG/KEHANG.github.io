---
layout: post
title:  "Demystifying Named Entity Recognition - Part II"
date:   2019-06-15 10:16:16
categories: explained
---

As a continuation for [Demystifying Named Entity Recognition - Part I]({% post_url 2019-05-14-named-entity-recognition %}), in this post I'll discuss popular models available in the field and try to cover:

- popular **traditional** models

- **deep learning** models

- **off-the-shelf** options

Over the history of [NER](https://en.wikipedia.org/wiki/Named-entity_recognition), there's been three major approaches: grammar-based, dictionary-based and machine-learning-based. 
Grammar-based approach produces a set of empirical rules hand-crafted by experienced computational linguists, usually takes months of work.
Dictionary-based approach basically organizes all the known entities into a lookup table, which can be used to detect whether a candidate belongs to a defined category or not. By design it doesn't work well with newly invented entities. 
Machine-learning-based approach typically needs annotated data, but doesn't necessarily rely on domain experts to come up with rules or fail on unseen entities.

This post focuses only on machine-learning based models. 

## 1. Popular traditional models

The traditional models we'll discuss here are [MEMM](https://en.wikipedia.org/wiki/Maximum-entropy_Markov_model), [CRF](https://en.wikipedia.org/wiki/Conditional_random_field). They are very popularly used before deep learning models entered the scene. 

### 1.1 MEMM

We've covered the details of MEMM in the [previous post]({% post_url 2019-05-14-named-entity-recognition %}). The key idea of the MEME approach is to model the **conditional probability** of tag sequenece for a given sentence with Markov assumption:

$$p(y_1...y_n | x_1...x_n) = \prod_{i=1}^{n} p(y_i |y_{i-1}, x_1...x_n)$$

We then model $$p(y_i | y_{i-1}, x_1...x_n)$$ 
using local environment:

$$p(y_i | y_{i-1}, x_1...x_n) = \frac{\exp({\underline{\theta} \cdot \underline{f}(y_{i-1}, y_i, x_1...x_n)})}{\sum_{y'}{\exp({\underline{\theta} \cdot \underline{f}(y_{i-1}, y', x_1...x_n)})}}$$

**In inference**, we use *Viterbi* algorithm to get best-fitting tag sequence for a given sentence. Details can be found in 2.2.1 section of the [previous post]({% post_url 2019-05-14-named-entity-recognition %}).

**In training**, we use *maximum likelihood estimation* to get optimal $$\underline{\theta}$$ that 

$$max_{\underline{\theta}} \prod_{j=1}^{N} p(\underline{x}^j | \underline{y}^j)$$

where $$\underline{x}^j, \underline{y}^j$$ are the $$j^{th}$$ sentence and corresponding tag sequence (the whole training dataset has $$N$$ examples).

### 1.2 CRF

Instead of $$p(y_i | y_{i-1}, \underline{x})$$
, 
Conditional Random Field (CRF) approach chooses to directly model $$p(\underline{y} | \underline{x})$$:

$$p(\underline{y} | \underline{x}) = \frac{\exp({\underline{\Theta} \cdot \underline{F}(\underline{x}, \underline{y})})}{\sum_{\underline{y}'}{\exp({\underline{\Theta} \cdot \underline{F}(\underline{x}, \underline{y}')})}}$$

The main challenge in direct modeling is that the denominator is sum of $$K^n$$ terms where $$K$$ is the number of tag label types and $$n$$ is the length of sentence to tag. This is a much larger number than that in MEMM - $$p(y_i | y_{i-1}, x_1...x_n)$$ 
has just $$K$$ terms in the denominator.

#### 1.2.1 Inference
During inference, we are only interested in the $$\underline{y}^{*}$$ that gives the highest probability rather than the highest probability itself:

$$
\underline{y}^{*} = \text{arg} \max_{\underline{y}} p(\underline{y} | \underline{x}) = \text{arg} \max_{\underline{y}}\exp({\underline{\Theta} \cdot \underline{F}(\underline{x}, \underline{y})}) = \text{arg} \max_{\underline{y}}{\underline{\Theta} \cdot \underline{F}(\underline{x}, \underline{y})}
$$

If using brutal force, we have to evaluate $$\exp({\underline{\Theta} \cdot \underline{F}(\underline{x}, \underline{y})})$$ for $$K^n$$ times. 

Fortunately, if we add a little structure into $$\underline{F}(\underline{x}, \underline{y})$$, which I'm going to talk about next, we can bring the exponential complexity - $$O(K^n)$$ down to linear complexity - $$O(K^2n)$$.

The structure added in CRF is:

$$\underline{F}(\underline{x}, \underline{y}) = \sum_{i=1}^n \underline{f}(\underline{x}, y_{i-1}, y_i)$$

To maximize $$\underline{\Theta} \cdot \underline{F}(\underline{x}, \underline{y})$$, we define a partial score as we did in 2.2.1 section of the [previous post]({% post_url 2019-05-14-named-entity-recognition %}):

$$
s_{partial, k}(y_{1...k}) = \underline{\Theta} \cdot \sum_{i=1}^k \underline{f}(\underline{x}, y_{i-1}, y_i)
$$ 

If we can maximize any partial score (which turns out not that difficult), then the score we want to acutally maximize, $$\underline{\Theta} \cdot \underline{F}(\underline{x}, \underline{y})$$, is just a special case of $$s_{partial, k}$$ when $$k=n$$.

So how to maximize any partial score? Let's start with $$k=1$$, namely $$s_{partial, 1} (y_1)= \underline{\Theta} \cdot \underline{f}(\underline{x}, y_1)$$.
> This is easy because it's just a single-variable optimization and $$y_1$$ can only have K choices. We also store all the evaluated $$s_{partial, 1} (y_1)$$.

How about $$k=2$$, namely maximize $$s_{partial, 2}(y_1, y_2) = s_{partial, 1}(y_1) + \underline{\Theta} \cdot \underline{f}(\underline{x}, y_1, y_2)$$?
> We can first fix $$y_2$$ and optimize over $$y_1$$ dimension. Remember we've known $$s_{partial, 1}(y_1)$$ evaluated from the previous question. So it takes $$K$$ computations to find the optimal $$y_1$$ for each $$y_2$$ - $$s_{partial, 2}(y_1^*, y_2)$$. Then pick the $$y_2^*$$ which has maximum $$s_{partial, 2}(y_1^*, y_2)$$. In total, we need perform $$K^2$$ evaluations. We also store all the $$s_{partial, 2}(y_1^*, y_2)$$ for future use.

How about $$k=3$$, namely maximize $$s_{partial, 3}(y_1, y_2, y_3) = s_{partial, 2}(y_1, y_2) + \underline{\Theta} \cdot \underline{f}(\underline{x}, y_2, y_3)$$?
> Similar to the previous question, we try to estimate $$s_{partial, 3}(y_1^*, y_2^*, y_3)$$ for each $$y_3$$ using
$$s_{partial, 3}(y_1^*, y_2^*, y_3) = \max_{y_2}(s_{partial, 2}(y_1^*, y_2) + \underline{\Theta} \cdot \underline{f}(\underline{x}, y_2, y_3))$$. We also carry out $$K$$ evaluation per $$y_3$$, thus totally $$K^2$$ evaluations for all possible $$y_3$$. We store $$s_{partial, 3}(y_1^*, y_2^*, y_3)$$ for future use (e.g., when $$k=4$$).

By doing this all the way to $$k=n$$, we can get $$\max_{\underline{y}}{\underline{\Theta} \cdot \underline{F}(\underline{x}, \underline{y})}$$ with roughly $$K^2n$$ evaluations.

#### 1.2.2 Training
Similar to MEMM, we can also use *maximum likelihood estimation* to get optimal $$\underline{\Theta}$$ that 

$$max_{\underline{\Theta}} \prod_{j=1}^{N} p(\underline{x}^j | \underline{y}^j)$$

where $$\underline{x}^j, \underline{y}^j$$ are the $$j^{th}$$ sentence and corresponding tag sequence (the whole training dataset has $$N$$ examples). More details on training algorithm can be found
in Page 10 of [Michael Collins's CRF note](http://www.cs.columbia.edu/~mcollins/crf.pdf).

## 2. Deep learning models

The deep learning models we'll discuss here are [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory), [BiLSTM-CRF](https://arxiv.org/abs/1508.01991), [Bert](https://arxiv.org/pdf/1810.04805.pdf). 

### 2.1 LSTM

### 2.2 BiLSTM-CRF

### 2.3 Bert

## 3. Off-the-shelf options
