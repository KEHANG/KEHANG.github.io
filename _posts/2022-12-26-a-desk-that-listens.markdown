---
layout: post
title:  "A Desk That Listens"
date:   2022-12-26 10:16:16
categories: iot
---

As a continuation of my last post [A Desk with Its Own Schedule]({% post_url 2022-11-25-a-desk-with-its-own-schedule %}), I'm building a new version of it; adding voice-control capability so that it listens to my commands.

One might think it's not really necessary to voice control a desk to go up and down since the buttons on the control panel already do these very easily. Two reasons for me to carry out this version:

- introducing a new communication mechanism (i.e., natural language) to my desk is technically very interesting; as you will see, it involves designing a whole series of modules that work together such as wake word detection, speech recognition, intent classification and execution.

- adding voice-control actually makes it possible to have more sophisticated desk behaviors. For instance, asking the desk to rest for 10 minutes and then go back to work again, or start a working schedule at 10 am on Monday.

Here is how it looks: [video demo](https://youtu.be/2v-3ZvoLdjo) ([code](https://github.com/KEHANG/smart-desk)).

<iframe width="100%" height="360" src="https://www.youtube.com/embed/2v-3ZvoLdjo" frameborder="0" allowfullscreen></iframe>


## Voice-control workflow

Just like Google Home Assistant, for voice control to work, we have to build a system of four components and coordinate them to work together.

{:.srs_img}
![Alt text]({{ site.github.url }}/assets/smart_desk_v2_post/img/workflow.png){:width="100%"}

- wake word detection module listens to the audio stream constantly and triggers speech recognition module when it picks up certain words (e.g., Hey Goolge).

- speech recognition module converts audio waveform into text representation.

- intent detection module takes in a text sentence and figures out the intent and associated parameters (e.g., `rest for 10 mins` can be classified as the intent of `going down` with argument `10 mins`).

- based on the intent, corresponding APIs can be used to acutally execute the actions.

I decided to use `Bob` as my wake word (so that becomes the name of my desk). Not to complicate things in the first try, I built the logic flow via the following code:

```python
# Check if the wake word is present in the speech
if is_wake_word(text, wake_word):
	# Wake word has been detected, do something...
	print("Wake word detected:", text)
	intent, kwargs = intent_detection.detect_intent(text)
	print("Intent detected:", intent, kwargs)
	execute(intent, **kwargs)
```

## Wake word detection

In this prototype, my detector is made extremely simple using library [Uberi/speech_recognition](https://github.com/Uberi/speech_recognition): check if `Bob` appears in the text sentence recognized by `sr.Recognizer()`. For advanced use cases, one may find this library useful [Picovoice/porcupine](https://github.com/Picovoice/porcupine).

## Speech recognition

[Uberi/speech_recognition](https://github.com/Uberi/speech_recognition) provides multiple APIs including `CMU Sphinx`, `Google Speech Recognition`, `Microsoft Azure Speech` etc. I used `Google Speech Recognition` in this post and found that very easy to hook up with. The library even provides a generic API key so if you don't use it too heavily you can call the API out of the box. If one feels like not to rely on those APIs from big tech, it's not a bad idea to build a speech recognizer from scratch via machine learning. Here is an example [speech recognition by Michael Phi](https://www.youtube.com/watch?v=YereI6Gn3bM).


## Intent detection

Intent detection is also a good place to utilize machine learning. Here again not to complicate things, I built a keyword-based solution to classify intents.

```python
# Define the keywords that indicate each intent
stand_up_keywords = ["stand", "up", "rest"]
sit_down_keywords = ["sit", "down", "work"]
report_height_keywords = ["height"]

# Check if any of the keywords for each intent appear in the text
if any(word in text for word in stand_up_keywords):
    intent = "rest"
    # Extract the time duration information from the command text
    kwargs = {"timeout": extract_duration(text)}
...
...
```


## Acknowledgements

I'd like to thank a couple of projects here:

- [Uberi/speech_recognition](https://github.com/Uberi/speech_recognition) provides the backbone of my voice-control system.
- This is the first time I used a machine/AI for pair-programming, yes you guessed right [chatGPT](https://chat.openai.com/chat), which certainly made the whole development process delightful.
