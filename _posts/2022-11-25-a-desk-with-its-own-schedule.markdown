---
layout: post
title:  "A Desk with Its Own Schedule"
date:   2022-11-25 10:16:16
categories: iot
---

Ever since I read Armin Ronarcher’s [post](https://lucumr.pocoo.org/2020/5/24/my-standard-desktop/) on how he controlled his desk with shell commands, I’ve been wanting to connect to my standing desk and add some intelligence to it. The immediate challenge is how to connect. Armin uses desk's bluetooth feature, but my [Flexispot desk](https://amzn.to/3Ez3rH6) doesn’t have this, so I approached it with with physical connection: linking a Raspberry Pi to my desk, then programming the Pi to introduce customized behaviour. 

I tend to forget to stand up and rest during work, so the first version presented here is to give a schedule to my desk. Here is how it looks: [video demo](https://youtu.be/lJAUM3Wqgdk) ([code](https://github.com/KEHANG/smart-desk)).

<iframe width="100%" height="360" src="https://www.youtube.com/embed/lJAUM3Wqgdk" frameborder="0" allowfullscreen></iframe>

## High level connection

A standing desk usually uses a control panel/pad to send commands (up, down, etc.) to the desk's motor controller via a RJ45 cable (Ethernet cable).

{:.srs_img}
![Alt text]({{ site.github.url }}/assets/smart_desk_post/img/connection.png){:width="100%"}

In my case, I'll replace the control panel with a Raspberry Pi (that way, we can implement some intelligence logic on the Pi side, which becomes the brain of our desk). How to connect? We can still use a RJ45 cable with an additional component: [RJ45 breakout board](https://amzn.to/3OGtDEt). The breakout board flattens the 8 pins of RJ45 in a way that’s super convenient to wire with Pi’s GPIO pins.

## Low level connection

Importantly we need to connect 4 pins of the RJ45 with 4 GPIO pins of the Pi. The exact mapping depends on the desk model. In my case, it's a [Flexispot EC3](https://amzn.to/3Ez3rH6) model which uses the HS11A-1 control panel. The information flows like this: command (in bytes format) gets emitted from the Pi's pin 8 (i.e., TX), passes through RJ45 pin 6 and arrives in motor controller's RX. data like desk height (in bytes format) get  emitted from the motor controller's TX, passes through RJ45 pin 5 and arrives in Pi's pin 10 (i.e., RX).

{:.srs_img}
![Alt text]({{ site.github.url }}/assets/smart_desk_post/img/low-level-connection.png){:width="100%"}

Note other models using different control panels may have RJ45 pins arranged differently. (let me know if you'd like some details on how to debug pins for other models)

## Command encoding

When commands are sent to controller, they have to be a stream of bytes. Thanks to [nv1t/standing-desk-interceptor](https://github.com/nv1t/standing-desk-interceptor) and [iMicknl/LoctekMotion_IoT](https://github.com/iMicknl/LoctekMotion_IoT), below is a mapping between the major commands and their bytes format. For instance, the `up` command is encoded as `\x9b\x06\x02\x01\x00\xfc\xa0\x9d`.

| Command name      | Start | Length | Type | Payload   | Checksum  | End  |
| ----------------- | ----- | ------ | ---- | --------- | --------- | ---- |
| `up`              | `9b`  | `06`   | `02` | `01` `00` | `fc` `a0` | `9d` |
| `down`            | `9b`  | `06`   | `02` | `02` `00` | `0c` `a0` | `9d` |
| `preset 1`        | `9b`  | `06`   | `02` | `04` `00` | `ac` `a3` | `9d` |
| `preset 2`        | `9b`  | `06`   | `02` | `08` `00` | `ac` `a6` | `9d` |
| `preset 3`        | `9b`  | `06`   | `02` | `10` `00` | `ac` `ac` | `9d` |

Note the `preset` commands make the desk go to the preset heights.

## Height data decoding

In reverse direction, height data gets sent from controller to the Pi's RX pin. The data is also encoded in bytes. Below is an example packet.

| Prefix           || Encoded height || Suffix 		   |
| ---------------- || -------------- || ---------------- |
| `\x9b\x07\x12`   || `\x5b\xff\x06` || `\x99\x24\x9d`   |


The encoded height has three byte characters, each representing a digit in the height. For instance, `\x06` → `1` (read futher on [why such mapping is made](https://alselectro.wordpress.com/2015/03/03/8051-tutorials-3-interfacing-7-segment-display/)). I'm listing the mapping of 10 digits to the bytes down below.


| Byte     | `\x3f` | `\x06` | `\x5b` | `\x4f` |`\x66` | `\x6d` |`\x7c` | `\x07` |`\x7f` | `\x6f` |
| ---------| ------ | ------ | ------ | ------ | ----- | -----  | ----- | ------ | ----- | ------ | 
| Digit    | `0`    | `1`    | `2`    | `3`    | `4`   | `5`    | `6`   | `7`    | `8`   | `9`    |

It's worth noting that a digit with a decimal point is encoded differently than a pure digit. So here's another mapping just for that.

| Byte     | `\xbf` | `\x86` | `\xdb` | `\xcf` |`\xe6` | `\xed` |`\xfc` | `\x87` |`\xff` | `\xef` |
| ---------| ------ | ------ | ------ | ------ | ----- | -----  | ----- | ------ | ----- | ------ | 
| Digit    | `0.`    | `1.`    | `2.`    | `3.`    | `4.`   | `5.`    | `6.`   | `7.`    | `8.`   | `9.`    |

Thus the above example of encoded height `\x5b\xff\x06` gets decoded to `28.1` inch.

## Scheduling

Based on my personal habit, I created [this scheduling script](https://github.com/KEHANG/smart-desk/blob/main/schedule.py) to split each hour for my standing desk: 50 min work + 10 min rest. Each day the desk runs 8 rounds of work and rest. A crontab job is scheduled to run this script on desired days.

## Acknowledgements

I'd like to thank several projects here:

- [LoctekMotion_IoT](https://github.com/iMicknl/LoctekMotion_IoT) has a very good README, summarizes findings from other projects and covers a lot of basics.
- I learned how to decode the height data from [LoctekReverseengineering](https://github.com/VinzSpring/LoctekReverseengineering).
- [alselectro](https://alselectro.wordpress.com/2015/03/03/8051-tutorials-3-interfacing-7-segment-display/) has a detailed and educational post on how a `digit` gets rendered via 7-segment display, basically why `1` is represented by `\x06`.
