---
layout: post
title: "Demystifying Named Entity Recognition - Part II"
date: 2019-06-15 10:16:16
categories: nlp, explained
---

As a continuation for [Demystifying Named Entity Recognition - Part I]({% post_url 2019-05-14-named-entity-recognition %}), in this post I'll discuss popular models available in the field and try to cover:

- popular **traditional** models

- **deep learning** models

- python libraries

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

**In inference**, we use _Viterbi_ algorithm to get best-fitting tag sequence for a given sentence. Details can be found in 2.2.1 section of the [previous post]({% post_url 2019-05-14-named-entity-recognition %}).

**In training**, we use _maximum likelihood estimation_ to get optimal $$\underline{\theta}$$ that

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
> $$s_{partial, 3}(y_1^*, y_2^*, y_3) = \max_{y_2}(s_{partial, 2}(y_1^*, y_2) + \underline{\Theta} \cdot \underline{f}(\underline{x}, y_2, y_3))$$. We also carry out $$K$$ evaluation per $$y_3$$, thus totally $$K^2$$ evaluations for all possible $$y_3$$. We store $$s_{partial, 3}(y_1^*, y_2^*, y_3)$$ for future use (e.g., when $$k=4$$).

By doing this all the way to $$k=n$$, we can get $$\max_{\underline{y}}{\underline{\Theta} \cdot \underline{F}(\underline{x}, \underline{y})}$$ with roughly $$K^2n$$ evaluations.

#### 1.2.2 Training

Similar to MEMM, we can also use _maximum likelihood estimation_ to get optimal $$\underline{\Theta}$$ that

$$max_{\underline{\Theta}} \prod_{j=1}^{N} p(\underline{x}^j | \underline{y}^j)$$

where $$\underline{x}^j, \underline{y}^j$$ are the $$j^{th}$$ sentence and corresponding tag sequence (the whole training dataset has $$N$$ examples). More details on training algorithm can be found
in Page 10 of [Michael Collins's CRF note](http://www.cs.columbia.edu/~mcollins/crf.pdf).

## 2. Deep learning models

The deep learning models we'll discuss here are [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory), [BiLSTM-CRF](https://arxiv.org/abs/1508.01991), [Bert](https://arxiv.org/pdf/1810.04805.pdf).

### 2.1 LSTM

#### 2.1.1 Architecture

{: .srs_img}
![Alt text]({{ site.github.url }}/assets/ner_post/img/lstm.png){:width="100%"}

In the setting of LSTM, each token $$x_i$$ is fed to a LSTM unit, which outputs a $$o_i$$. $$o_i$$ models log probabilities of all possible tags at i-th position, so it has dimension of $$K$$.

$$o_i = \begin{bmatrix}log P(y_i = PER | \underline{x})\\log P(y_i = ORG | \underline{x})\\...\\log P(y_i = MISC | \underline{x})\end{bmatrix}$$

#### 2.1.2 Inference

The inference in LSTM is very simple: $$y_i$$ = the tag with highest log probability at i-th position.

$$y_i^* = argmax_k o_{i,k}$$

which indicates the prediction of i-th position only utilizes the sentence information up to i-th token - only the left side of the sentence is used for tag prediction at i-th position. BiLSTM is designed to provide context information from both sides, which will be seen in next section.

#### 2.1.3 Training

Like all the other neural network training, LSTM training uses **Stochastic Gradient Descent** algorithm. Loss function adopts **negative log likelihood**. For a data point $$(\underline{x^j}, \underline{y^j})$$, we have its loss calculated as:

$$L_j = -\sum_{i=1}^{n_j} o_i^j[y_{i}^j]$$

where $$n_j$$ is the length of the sentence $$x^j$$, $$o_i^j$$ is the LSTM output at i-th position and $$y_i^j$$ is the ground truth tag at i-th position.

**Total loss** is the mean of all the individual losses.

$$L = \frac{1}{N}\sum_{j=1}^N L_j$$

where $$N$$ is the total number of training examples.

### 2.2 BiLSTM

{: .srs_img}
![Alt text]({{ site.github.url }}/assets/ner_post/img/bilstm.png){:width="120%"}

BiLSTM stands for bi-directional LSTM, which provides sequence information from both directions. Because of that, BiLSTM is more powerful than LSTM. Except the bi-directional component, the meaning of network output, inference, and training loss are same as LSTM.

### 2.3 BiLSTM-CRF

BiLSTM captures contextual information around i-th position. But at each position, BiLSTM predicts tags basically in an independent fashion. There's cases where some adjacent positions are predicted with tags which do not usually appear together in reality. For example, I-PER tag should not follow B-ORG. To account for this kind of interactions between adjacent tags, Conditional Random Field (CRF) is introduced to BiLSTM.

#### 2.3.1 Architecture

{: .srs_img}
![Alt text]({{ site.github.url }}/assets/ner_post/img/bilstm-crf.png){:width="120%"}

where $$o_i$$ models **emission scores** of all possible tags at i-th position and $$y_i^*$$ is the best tag for i-th position which collectively achieves highest sequence score.

$$o_i = \begin{bmatrix} score_{emission}(y_i = PER | \underline{x})\\score_{emission}(y_i = ORG | \underline{x})\\...\\score_{emission}(y_i = MISC | \underline{x})\end{bmatrix}$$

CRF layer also learns a transition matrix $$A$$ which stores transition scores between any possible pair of tag types.

#### 2.3.1 Inference

Same as the inference in CRF section, given a trained network and sentence $$\underline{x}$$, any sequence $$\underline{s}$$ will have a score.

$$score(\underline{x}, \underline{s}) = \sum_{i=1}^n o_i[s_i] + A[s_{i-1}][s_i]= \sum_{i=1}^n \phi(\underline{x}, s_{i-1}, s_i)$$

The score is a sum of contributions from token level. i-th position has contribution of $$\phi(\underline{x}, s_{i-1}, s_i) = o_i[s_i] + A[s_{i-1}][s_i]$$, where the first term is emission score and second term is transition score.

To find the tag sequence $$\underline{y}^*$$ achieving highest score, we need to use dynamic programming.

Define sub problem $$DP(k,t)$$ to be the max score accumulated from 1st position to $$k$$-th position with the $$k$$-th position tag being $$t$$, detailed as follows:

$$DP(k, t) = \max \limits_{\underline{s}\in S^k:s_k=t} \sum_{i=1}^k \phi(\underline{x}, s_{i-1}, s_i)$$

The recursion would be:

$$DP(k+1, t) = \max \limits_{t'} [DP(k, t') + \phi(\underline{x}, t', t)]$$

The original problem is then

$$score(\underline{x}, \underline{y}^*) = \max \limits_{t} DP(n, t)$$

We can always use [parent pointers](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/MIT6_006F11_lec20.pdf) to retrieve the corresponding best sequence $$\underline{y}^*$$.

#### 2.3.2 Training

Loss function for BiLSTM-CRF also adopts **negative log likelihood**. For a data point $$(\underline{x^j}, \underline{y^j})$$, we have its loss calculated as:

$$L_j = -log P(\underline{y}^j | \underline{x}^j) = - log \frac{exp(score(\underline{x}^j, \underline{y}^j))}{\sum \limits_{\underline{y'}^j} exp(score(\underline{x}^j, \underline{y'}^j))}$$

$$ = - score(\underline{x}^j, \underline{y}^j) + log \sum \limits\_{\underline{y'}^j} exp(score(\underline{x}^j, \underline{y'}^j))$$

where the first term is easy to calculate via a forward pass of the network and the second term needs more care. Let's define that term (without log) as $$Z$$, which is exponential sum of scores of all the possible sequences $$\underline{s}$$ of length $$n$$.

$$Z = \sum \limits_{\underline{s} \in S^n} exp(score(\underline{x}, \underline{s})) = \sum \limits_{\underline{s} \in S^n} exp(\sum_{i=1}^{n} \phi(\underline{x}, s_{i-1}, s_i))$$

$$= \sum \limits_{\underline{s} \in S^n} \prod_{i=1}^{n} exp(\phi(\underline{x}, s_{i-1}, s_i)) = \sum \limits_{\underline{s} \in S^n} \prod_{i=1}^{n} \psi(\underline{x}, s_{i-1}, s_i) $$

To calculate $$Z$$, we need to use dynamic programming again. This time the sub-problem $$DP(k,t)$$ is the exponential sum of scores of all possible sequences of length $$k$$ with last tag $$s_k = t$$:

$$ DP(k,t)= \sum \limits*{\underline{s} \in S^k: s_k=t} \prod*{i=1}^{k} \psi(\underline{x}, s\_{i-1}, s_i) $$

The recursion would be:

$$ DP(k+1,t) = \sum \limits\_{t'} DP(k,t')\cdot \psi(\underline{x}, t', t)$$

The original problem is then

$$Z = \sum \limits_{t} DP(n,t)$$

Via this way, individual loss $$L_j$$ is calculated and then batch loss by averaging the individual losses in the batch.

### 2.4 Bert

Recent research on BERT provides an option for NER modeling. Despite of the complexity of the BERT model architecture, in the context of NER it can be regarded as an advanced version of our BiLSTM model - replacing the LSTM with multiple [Transformer Encoder](http://jalammar.github.io/illustrated-transformer/) layers.

{: .srs_img}
![Alt text]({{ site.github.url }}/assets/ner_post/img/bert.png){:width="120%"}

Thus, $$o_i$$ still models log probabilities of all possible tags at i-th position.

$$o_i = \begin{bmatrix}log P(y_i = PER | \underline{x})\\log P(y_i = ORG | \underline{x})\\...\\log P(y_i = MISC | \underline{x})\end{bmatrix}$$

Inference, and training loss are same as LSTM section.

## 3. Python libraries

There's several machine learning based NER repositories in GitHub. I picked some of them here with some comments.

- [KEHANG/ner](https://github.com/KEHANG/ner/): for English texts, based on PyTorch, has LSTM, BiLSTM, BiLSTM+CRF and Bert models, has released conda package

- [shiyybua/NER](https://github.com/shiyybua/NER): for Chinese texts, based on Tensorflow, only BiLSTM+CRF model, no packages released

- [Franck-Dernoncourt/NeuroNER](https://github.com/Franck-Dernoncourt/NeuroNER): for English texts, based on Tensorflow, has LSTM model, no package released
