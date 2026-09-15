# miniLM 0.41M
a mini generative language model, built from scratch.
---
## architecture: Decoder Transformer
* total number of layers `14`
* multiheadAttention blocks `2`
* feedforwardNetwork `2`
* Non-Linearities: `Tanh, ReLU`
<!--image section-->

---
### this model was trained directly in one stage of training which was pretraining with dataset like finetunning dataset.
as the dataset was conversations, exactly after training, the model could keep up in live conversation, yet with broken
words formatting and gramma errors!
