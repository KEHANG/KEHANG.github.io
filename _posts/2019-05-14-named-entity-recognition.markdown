---
layout: post
title:  "Named Entity Recognition Explained"
date:   2019-05-14 17:16:16
categories: explained
---

Recently I've been working on a project related to **Named Entity Recognition** (**NER**). At the very beginning, I was trying to find a well-explained document to get myself started, but couldn't do so (instead I found redundant pieces here and there on the Internet). My requirement is simple. It should include

- *what* is **NER**

- *how* to **formulate** it

- *what* are the **traditional** and **start-of-the-art** models

- *what* are the **off-the-shelf** options

- *how* to build a **customized** NER model

So this post will try to provide a complete set of explanation on these questions.


### 1. What is NER

Simply put, **Named Entity Recognition** is a technology that is able to identify **certain entities** from a sentence/paragraph/document. Like the sentence below, **NER** tries to tag each word with a label; `Steve Jobs` and `Steve Wozniak` to be `PER` (persion entity), `Apple` to be `ORG` (organization entity) and the rest `O` (not an entity).


{: .srs_img}
![Alt text]({{ site.github.url }}/assets/ner_post/img/what-is-ner.png){:width="100%"}


This is useful in lots of cases. For instance, *Apple Mail* identifies `TIME` entity in emails and makes pulling events from email to calendar much easier than before; *Google Search* is able to find `Headquarters` entity in a relevent document to answer query questions.

{: .srs_img}
![Alt text]({{ site.github.url }}/assets/ner_post/img/what-is-ner-2.png){:width="100%"}

### 2. Mathematic Formulation

### 3. Traditional Models

### 4. Start-of-the-Art

### 5. Off-the-Shelf Options

### 6. Build Your Own Model