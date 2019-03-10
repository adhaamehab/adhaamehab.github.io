### LSTM for arabic part-of-speech tagging

### Part1: Introduction

Hello, In this post I'm going to explain my understanding of ***LSTM*** Network and How I used it to build an Arabic part of speech tagger based on Universal Dependancy Tree Bank. This post is part of a series in building a python package for Arabic natural language processing. You can check the previous post [here](https://adhaamehab.me/2019/02/01/gp-docs.html)

In general, Neural Networks are pattern recognition models which learns and enhance by iterating over dataset and get better in recognising the pattern within a data. 

Recurrent Neural Network is designed to prevent neural networks from decay by using feedback loops. Those feedback loops are what makes RNN better at solving the sequence learning task.

LSTM is the best RNN architecture for the *sequence learning task* for using memory of past input.

The Human brain think persistently. For example, while you are reading this post you don't throw everything away and start thinking from scratch when you are reading a new word. In fact you do understand each word based on the previous words.

Traditional neural networks (*Feed forward networks*) aren't able to do this as they lack reasoning.



### Part2: Recurrent nerual networks

RNNs solve this issue by using memory units. The output that the RNN produce at step $N$ is affected by the output from the step $N -1$. So in general RNN has two sources of inpu. One is the actual input and two is the context (memory) unit from previous input.

```
That sequential information is preserved in the recurrent network’s hidden state, which manages to span many time steps as it cascades forward to affect the processing of each new example. It is finding correlations between events separated by many moments, and these correlations are called “long-term dependencies”, because an event downstream in time depends upon, and is a function of, one or more events that came before. One way to think about RNNs is this: they are a way to share weights over time. [1]
```

Mathematically,

*Feed forwared* networks are defined by the formula $H_t = F(W * X)​$.

The new state is a function of the *weights matrix*  multiplied by the *input vector*

This method is called *Activation function*.

The same process are applied for *Recurrent network* with a simple modification.

$H_t = F(W * X + U * H_{t-1})$ The previous state/result is first multiplied by a matrix $U$ 

called *hiden-state-to-hiden-state* matrix then added to the input of the activation function

So $H_t$ won't be only affected by by $H_{t-1}$ but all the prevoius hiden states that affected $H_{t-1}$ which will insute the presistent of memory.

### Part3: Long Short-Term Memory Architecture

RNNs are great. They helped alot in solving many tasks in NLP. Still they have the problem of Long-Term Dependencies.

> you can read more about LTD in Colah's great blog post [here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

LSTMs are desinged to solve the LTD issue by remembering information for long time

The only difference beteween RNN and LSTM that instead having a single neural network layer in RNN. We have 4 layers in LSTM interacting together in a special way.

### Part4: Part-of-speech tagging

Part of speech tagging is the task of labeling each word in a sentence with a grammatical tag that define the  grammatical tagging or word-category disambiguation of the word in this sentence.

The problem here is to determine the POS tag for a particular instance of a word within a sentance.

This tags can be used to solve more advanced problems in ***NLP*** like 

- Language Understanding as, Knowing the tags means helps in obtaining a better understanding of text as different words can have different meaning based on their location in the sentence and its neighbors words.
- Text-to-speech and automated speaking tone control.

While for english language PoS tagging is an already-solved-problem. For a reach morphological language like Arabic. The problem still presists and there is ***ZERO*** open source deep-learning based arabic part-of-speech tagger.

### Part5: Universal dependancy Tree Bank

The lack of open source tagger is an obvious result of the lack of an arabic treebank dataset.

The well-known ***Penn Tree Bank*** costs around ***3k$*** and the ***Quranic Tree bank*** is very classical and perform poorly on day-to-day words.

[Universal Dependencies](https://universaldependencies.org) (UD) is a framework for cross-linguistically consistent grammatical annotation and an open community effort with over 200 contributors producing more than 100 treebanks in over 70 languages Including ***Arabic***

UD provide 3 different treebanks, I used the best-reviewed one for building the model.

### Part6: Data Preprocessing

UD provides the data set in [CoNLL](http://universaldependencies.org/conll17/) form. I used [pyconll](https://github.com/pyconll/pyconll) to convert the dataset from CoNLL format into pandas data frame.

The desired structure of the data was like this 

```
[
    [(Word1, T1), (Word2, T2), ...., (WordN, TM)], # Sentence one
    [(Word1, T1), (Word2, T2), ...., (WordN, TM)], # Sentence two
    .
    .
    .
    [(Word1, T1), (Word2, T2), ...., (WordN, TM)] # Sentence M
]
```

 After converting the data set into the desired shape. I had 3 tasks to do with data to prepare it for keras to use it.

1 - WordEmbedding representation for sentences and tags.

2 - Padding every sentences so all sentences have the same size.

Word embedding is a way to represent raw textual data with unique integers so we can fed them to Neural networks. I used a very simple approach to do this by using a map between each word and its integer value.

So before implementation we had to maps `word2index` and `tag2index`



Because we are doing seq2seq task.The data should be transformed such that each sequence has the same length. This vectorization allows the LSTM model to efficiently perform batch matrix operation.

Also this will need a customized accuracy function that ignore the padding value as those values will always be predicted by the model.

### Part7: Keras implementation of LSTM

After converting the data to a suitable shape. The next step was to design and implement the actual model. 

#### The model

```Python
model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH, )))
model.add(Embedding(len(word2index), 128))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))
 
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy', ignore_class_accuracy(0)])
model.summary()
```

```shell
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_3 (Embedding)      (None, 398, 128)          3016192   
_________________________________________________________________
bidirectional_3 (Bidirection (None, 398, 512)          788480    
_________________________________________________________________
time_distributed_3 (TimeDist (None, 398, 17)           8721      
_________________________________________________________________
activation_3 (Activation)    (None, 398, 17)           0         
=================================================================
Total params: 3,813,393
Trainable params: 3,813,393
Non-trainable params: 0
_________________________________________________________________
```

Now let's explain the model thoroughly,

`InputLayer` is the first layer of the model. Keras has fixed size layer as explained in the preprocessing part so we define this layer with the maximum length of a sequence in the training set.

`Embedding` The embedding layer requires that the input data be integer encoded, so that each word is represented by a unique integer. It is intialized with random weights then it will learn for each word in the dataset

`LSTM`  The LSTM encoder layer is a sequence-to-sequence layer. It provides a sequence output rather than an integer value. And `return_sequences` force the layer to make the previous sequence an input for the next one.

`Bidirectional` Bidirectional wrapper duplicates the LSTM layer so we habe two side-by-side layers that transfers the resulted sequences to the inputed one. In practical this approach has a great effect on the long short-term memory. I used the default merge mode [concat] for the `Bidirectional` layer.

`TimeDistributed Dense` layer is used to keep one-to-one relations on input and output layers. It applies a same Dense (fully-connected) operation to every timestep of a 3D tensor.

`Activation` The activation method for the `Dense` layer. We could've defined the activation method as a paramater in the `Dense` layer but the second one is a better approach[*](https://stackoverflow.com/a/40870126)

We use `categorical_crossentropy` as a loss function because we have a *many-to-many labeling* problem.

and `Adam` optimizer (*adaptive moment estimation*) for training. 

`ignore_class_accuracy` is a method to recompute accuracy after ignoring the padding `<PAD>` 

### Part8: Results

The training steps took around 40 mins on my MacBook Pro. After the training we start evaluate the probelm 

```python
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', ignore_class_accuracy(0)])
model.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)))
```

```shell
680/680 [==============================] - 10s 14ms/step
[0.0009350364973001621, 0.0011567071230862947, 0.9160830103105271]
```

The model has reached `0.916` and started to converge after `30 epochs`

In the evaluation we used `stochastic gradient descnet` as it perform better than `Adam` in the evaluation[*](https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/)



Model Accuracy

![image-20190310185751465](/Users/adhaamehab/workspace/personal-blog/adhaamehab.me/minima/assets/images/accuracy.png)

Model Loss

![image-20190310185916986](/Users/adhaamehab/workspace/personal-blog/adhaamehab.me/minima/assets/images/loss.png)

Despite the results looks good comparing to stanford corenlp model still it can be enhanced. But as a first try it's not bad.

The whole source code is in the the project repository: https://github.com/adhaamehab/arabicnlp

### Part9: References

- https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/

- https://machinelearningmastery.com/data-preparation-variable-length-input-sequences-sequence-prediction/

- http://colah.github.io/posts/2015-08-Understanding-LSTMs/

- https://skymind.ai/wiki/lstm

- https://kevinzakka.github.io/2017/07/20/rnn/

- https://developer.nvidia.com/discover/lstm

- https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/

- http://karpathy.github.io/2015/05/21/rnn-effectiveness/

- https://universaldependencies.org/

  







