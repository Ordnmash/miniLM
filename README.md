# miniLM 8M
a mini generative language model, built from scratch.
---
## architecture: Decoder Transformer
* total number of layers `6`
* multiheadAttention blocks `1`
* feedforwardNetwork `2`
* Non-Linearities: `Tanh, GeLU`
<!--image section -->
<p align="center">
  <img src="https://t3.ftcdn.net/jpg/19/97/38/62/360_F_1997386203_neJiSmyUkCyVSPKu23t4G1RMNAn75pW4.jpg" width="600" height="400" alt="MiniLM">
</p>

---
### real-Chat sample from the model:
You:  Can we talk for a minute? 

    AI_friend:  That is whether we had a problem here. Is there something wrong with my son. 

You:  I don't know about that, is your son okay? 

    AI_friend:  I think I might be not bit. 

You:  Where is he? 

    AI_friend:  Go straight up to you? 

You:  Just tell me where he is, so I could check on him. 

    AI_friend:  By the way! 

You:  I hope everything is alright! 

    AI_friend:  Where is to do? 

You:  I don't know, you tell me. 

    AI_friend:  No problem, I liked about? 

You:  What do you like? 

    AI_friend:  It's very fond of you, but it is great food of card too romantic you understand on your dom tooth. It means great. 

You:  Okay, thank you for telling me that! 

    AI_friend:  What about my nose ask you some question. 

You:  Okay, go on ask me. 

    AI_friend:  Yes. You're really rating. 

You:  sure, Thank you! 

    AI_friend:  Yes, I am thinking about? How did you wait. Do you have a Trauff and they're coming. 

You:  I don't have that! 

    AI_friend:  Thank you. 

---
input tkns: 258 | model tkns: 3, indicates for history of chat and last model response.
