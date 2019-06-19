---
layout: post
title:  "Demystifying Named Entity Recognition - Part II"
date:   2019-06-15 10:16:16
categories: explained
---

As a continuation for [Demystifying Named Entity Recognition - Part I]({% post_url 2019-05-14-named-entity-recognition %}), in this post I'll discuss popular models available in the field and try to cover:

- popular **traditional** models

- **start-of-the-art** models

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

$$p(y_i | y_{i-1}, x_1...x_n) = \frac{\exp({\vec{\theta} \cdot \vec{f}(y_{i-1}, y_i, x_1...x_n)})}{\sum_{y'}{\exp({\vec{\theta} \cdot \vec{f}(y_{i-1}, y', x_1...x_n)})}}$$

**In training**, we use *maximum likelihood estimation* to get optimal $$\vec{\theta}$$ that 

$$max_{\theta} \prod_{j=1}^{N} p(y^j|x^j)$$

where $$x^j, y^j$$ are the $$j^{th}$$ sentence and corresponding tag sequence (the whole training dataset has $$N$$ examples).

**In inference**, we use *Viterbi* algorithm to get best-fitting tag sequence for a given sentence. Details can be found in 2.2.1 section of the [previous post]({% post_url 2019-05-14-named-entity-recognition %}).

### 1.2 CRF

## 2. State-of-the-art models

The state-of-the-art models we'll discuss here are [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory), [BiLSTM-CRF](https://arxiv.org/abs/1508.01991), [Bert](https://arxiv.org/pdf/1810.04805.pdf). 

### 2.1 LSTM

### 2.2 BiLSTM-

### 2.3 Bert

## 3. Off-the-shelf options
