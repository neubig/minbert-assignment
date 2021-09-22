# Structure of BertHW

## bert.py
This file contains the BERT Model whose backbone is the [transformer](https://arxiv.org/pdf/1706.03762.pdf). We recommend walking through Section 3 of the paper to understand each component of the transformer. 

### BertSelfAttention
The multi-head attention layer of the transformer. This layer maps a query and a set of key-value pairs to an output. The output is calculated as the weighted sum of the values, where the weight of each value is computed by a function that takes the query and the corresponding key. To implement this layer, you can:
1. linearly project the queries, keys, and values with their corresponding linear layers
2. split the vectors for multi-head attention
3. follow the equation to compute the attended output of each head
4. concatenate multi-head attention outputs to recover the original shape

<img src="https://render.githubusercontent.com/render/math?math=Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}}V)">

### BertLayer
This corresponds to one transformer layer which has 
1. a multi-head attention layer
2. add-norm layer
3. a feed-forward layer
4. another add-norm layer

### BertModel
This is the BertModel that takes the input ids and returns the contextualized representation for each word. The structure of the ```BertModel``` is:
1. an embedding layer that consists of word embedding ```word_embedding``` and positional embedding```pos_embedding```.
2. bert encoder layer which is a stack of ```config.num_hidden_layers``` ```BertLayer```
3. a projection layer for [CLS] token which is often used for classification tasks

The desired outputs are
1. ```last_hidden_state```: the contextualized embedding for each word of the sentence, taken from the last BertLayer (i.e. the output of the bert encoder)
2. ```pooler_output```: the [CLS] token embedding

### To be implemented
Components that require your implementations are comment with ```#todo```. The detailed instructions can be found in their corresponding code blocks
* ```bert.BertSelfAttention.attention``` 
* ```bert.BertLayer.add_norm```
* ```bert.BertLayer.forward```
* ```bert.BertModel.embed```

*ATTENTION:* you are free to re-organize the functions inside each class, but please don't change the variable names that correspond to BERT parameters. The change to these variable names will fail to load the pre-trained weights.

### Sanity check
We provide a sanity check function at sanity_check.py to test your implementation. It will reload two embeddings we computed with our reference implementation and check whether your implementation outputs match ours. 


## classifier.py
This file contains the pipeline to 
* call the BERT model to encode the sentences for their contextualized representations
* feed in the encoded representations for the sentence classification task
* fine-tune the Bert model on the downstream tasks (e.g. sentence classification)


### BertSentClassifier (to be implemented)
This class is used to
* encode the sentences using BERT to obtain the pooled output representation of the sentence.
* classify the sentence by applying dropout to the pooled-output and project it using a linear layer.
* adjust the model paramters depending on whether we are pre-training or fine-tuning BERT

## optimizer.py  (to be implemented)
This is where `AdamW` is defined.
You will need to update the `step()` function based on [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) and [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
There are a few slight variations on AdamW, pleae note the following:
- The reference uses the "efficient" method of computing the bias correction mentioned at the end of "2 Algorithm" in Kigma & Ba (2014) in place of the intermediate m hat and v hat method.
- The learning rate is incorporated into the weight decay update (unlike Loshchiloc & Hutter (2017)).
- There is no learning rate schedule.

You can check your optimizer implementation using `optimizer_test.py`.


## base_bert.py
This is the base class for the BertModel. It contains functions to 
1. initialize the weights ``init_weights``, ``_init_weights``
2. restore pre-trained weights ``from_pretrained``. Since we are using the weights from HuggingFace, we are doing a few mappings to match the parameter names
You won't need to modify this file in this assignment.

## tokenier.py
This is where `BertTokenizer` is defined. You won't need to modify this file in this assignment.

## config.py
This is where the configuration class is defined. You won't need to modify this file in this assignment.

## utils.py
This file contains utility functions for various purpose. You won't need to modify this file in this assignment.
 
## Reference
[Vaswani el at. + 2017] Attention is all you need https://arxiv.org/pdf/1706.03762.pdf
