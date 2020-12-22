## Structured Perceptron

A Julia implementation of the Structured Perceptron. 

The term structured perceptron describes (supervised) machine learning techniques that aim to predict structured objects.
Tasks of this kind are especially common in Natural Language Processing, but can also be found in other domains including Bioinformatics and Computer Vision. 

In computational linguistics, in particular, there exists a class of problems referred to as _sequence tagging_ e.g. part-of-speech (POS) tagging or named entity recognition (NER).
In the former case, for instance, each word in a sequence must receive a class label to describe its grammatical type:

e.g. \
_Input data:_ "Attack was the best option" \
_Labels:_ ['NOUN','VERB','DET','ADJ','NOUN]

We can see that even in this simple case there is already some ambiguity: in English "attack" can be a noun but also a verb! 

Classifying individual tokens will not take into account the fact that POS tags do not occur independently in a sequence, but instead each depends strongly on the ones around it. This feature can be nicely exploited in a sequence model, such as a Hidden Markov Movel, to predict the entire tag sequence instead of individual tags. 

Collins (2002) proposal of the structured perceptron combines the perceptron algorithm for learning linear classifiers with an inference algorithm that generates the candidate predictions.\
In this project, I implemented two different inference algorithms:
- the Viterbi algorithm, as in the original paper of Collins
- greedy search, in an way which is close to the [Brill tagger](https://en.wikipedia.org/wiki/Brill_tagger)

It is important to note that structured perceptrons are not necessarily the best possible solution (in terms of efficiency and/or performance) to this problem, but they possess some nice qualities. For instance, compared to using a neural network (e.g. RNNs or LSTMs), the structured perceptron can be trained very successfully on small datasets and it can give back nicely interpretable parameters. Good results may also be achieved using other popular machine learning models (e.g. structured SVMs). 


### Main Modules
---
#### SP_reader
This is a helper module, that contains functions for reading and pre-processing the data. I kept these separate from the main module to improve readability and keep the code cleaner. 

- `read_conll_file(file_name)` reads in the data from a CONLL file, which is the standard input for this kind of task. Data in this format are stored as following:
```julia
    word1    tag1
    ...      ...
    wordN    tagN
```
- `normalize!(word)` performs basic pre-processing, in place, on an input word. It lowercases everything, and performs regex substitutions of usernames, number and urls. 


#### SP
In this module, I define my Structured Perceptron object together with all functions for feature extraction, training, prediction, evaluation and saving/loading trained models. 

The **Structured Perceptron** is a mutable struct, which is initialized with default parameters before training.
It has the following attributes:
- `tags`: a set of strings, which is to include the possible labels for our structured prediction task. Initialized as empty
- `feature_weights`: a nested dictionary, initialized as empty. Given a feature, it is meant to map to it the weights associated to different possible tags. After a couple of iterations of training, this is an example of the weights it could return, for the feature indicating that the following word is "to":
```julia
julia> sp.feature_weights["NEXT_WORD=to"]
Dict{AbstractString,Float64} with 11 entries:
  "CONJ" => -0.25
  "DET"  => -0.5
  "ADJ"  => 0.25
  "NOUN" => 0.5
  "."    => 0.0
  "NUM"  => -0.25
  "VERB" => 0.0
  "ADV"  => 0.0
  "PRT"  => 0.75
  "PRON" => -0.25
  "ADP"  => -0.25 
```

- `tag_dict`: a default dictionary, initialised with empty sets as values. It is needed to map each normalized word in training to the set of allowed tags for that word, based on what the model learns. For instance, these are the tags associated to "that" based on on our training data.
```julia
julia> sp.tag_dict["that"]
Set{AbstractString} with 3 elements:
  "DET"
  "ADV"
  "ADP"
```

- `start_` and `end_` are just tokens that are added as padding to tokenized sentences before feature extraction and prediction. They are initialized as "_START_" and "_END_". The user can change them as he/she wishes, although this does not really affect performance.

The functions available in this module are the following:
- `fit!(sp,file_name,iterations,prediction_method;learning_rate,verbose)`: performs the training of the perceptron *sp*. It reads data from *file_name* using the function from SP_reader.jl, performs predictions with the method of choice and compares the predicted tags with the true tags to update `sp.feature_weights`. The learning rate and and option for printing information on training progress (*verbose*) are set as keyword arguments with default values 0.25 and "true" respectively. 
```julia
julia> SP.fit!(sp,file,epochs,prediction_method,learning_rate=0.25,verbose=true) 
***************
Iteration 1
.................................................1000
.................................................2000
203482 features
Training accuracy: 0.94
```

- `get_features(word,previous_tag2,previous_tag,words,i)`: computes a set of pre-determined features, for a given word and returns them as an array. In the case here, there are 22 features, which depend on the two previous tags (since we use a 2nd order Hidden Markov Model) and the surrounding words, plus a bias term. *i* is used when calling it in other functions, and it keeps track of the position word in the context (which is made up of the whole sentence plus start and end tokens). Current features are:
```julia
["PREFIX=","SUFFIX=","LEN<=3=", "WORD=","NORM_WORD=","PREV_WORD=",
"PREV_WORD_PREFIX=","PREV_WORD_SUFFIX=","PREV_WORD+WORD=", "NEXT_WORD=",
"NEXT_WORD_PREFIX=","NEXT_WORD_SUFFIX=","WORD+NEXT_WORD=","NEXT_2WORDS=",
"PREV_TAG=","PREV_TAG_BIGRAM=","PREV_TAG+WORD=","PREV_TAG+SUFFIX=",
"PREV_TAG+PREFIX=","WORD+TAG_BIGRAM=","SUFFIX+2TAGS=","PREFIX+2TAGS=","BIAS="]
 ```

- `get_global_features(sp,words,predicted_tags,true_tags)`: sums up local, word-wise features over the entire sentence, aggregating counts for each feature over the true tags and the predicted tags. Results are returned as defaultdicts, mapping tags to arrays of (feature,count) tuples. 
- `get_scores(sp,features)`: computes scores for each tag, given an array of features, using the current weights of the perceptron. Scores are returned as a defaultdict with tags as keys and scores as values. 
- `predict(sp,words,prediction_method)`: returns tag predictions for a sentence (*words*) based on the method of choice, using current parameter values for the perceptron. Predicted tags are returned in an array. If the prediction_method is "viterbi", it performs a call to `predict_viterbi()`, which uses Viterbi scoring for the inference. If instead we pick "greedy", which is the only other allowed option, it calls `predict_greedy()`, that performs a greedy search. This is an example of what a trained structured perceptron could generate when predicting on a sentence, noting that it must be tokenized beforehand:
```julia
julia> SP.predict(sp,tokenize.("This is my final project for this course."), "greedy")
9-element Array{AbstractString,1}:
 "DET"
 "VERB"
 "PRON"
 "ADJ"
 "NOUN"
 "ADP"
 "DET"
 "NOUN"
 "."
```

- `evaluate(sp,test_file,prediction_method)`: after reading in a data file, again in ConLL format, it performs predictions for each sentence using the structured perceptron and the preferred method, then returns overall accuracy. 
- `save_model(sp,out_name)`: saves the perceptron to a file as a Julia object, in .jld2 format. 
- `load_model(in_file)`: loads a previously saved perceptron object from a .jld2 file. One just need to assign it to a variable and Julia should automatically recognize its type, as long as **SP.jl** has been included. 

### Efficiency Considerations
---
Some optimization steps I took were the following:
- in the dynamic programming part of `predict_viterbi()` I build the scoring and backpointers matrices in a way that I could mostly loop by columns and I used  *@inbounds* to by-pass unnecessary bounds-checking, after making sure that everything was working correctly.
- while originally `read_conll_file()` returned nested arrays, I later switched to channels which allowed for gains in efficiency and far cleaner code. Here, although this function is currently not an issue, I believe it might be possible to get further improvements by better exploiting channels or using the Lazy.jl package (which I tried briefly, but could not make it work as intended).
- In general, I tried to use sets or tuples whenever possible, as an alternative to arrays, and defaultdicts instead of dicts to avoid get! calls.

Unfortunately, the last point was not always feasible. This in particular was the case for the weight updating procedure in `fit!()`. All of my attempts to use a defaultdict for `sp.feature_weights` or otherwise avoid the nested `get!(get!(...)..)` calls were unsuccessful. Indeed, inplace updating of values over nested dicts and defaultdicts via nested loops, generated unexpected behaviour in Julia (unlike Python) causing weights to be updated across all key combinations and breaking the entire model. Another option I could think of was to change data types, avoiding nested structures. This is what I had to do to solve the equivalent issue I was getting when updating counts in `get_global_features()`: the current solution is perhaps not the most elegant but it is not unefficient and, most importantly, it does the job. \
In the case of `sp.feature_weights`, doing this would not allow me to preserve the current nicely accessible and interpretable structure and it would over-complicate the code without many efficiency gains. Unfortunately, this part of the `fit()` function is currently a bit of a bottleneck. 

Another minor optimization issue is in `get_scores()`, which especially impacts `predict_viterbi()` since it is called very often during the Viterbi scoring procedure. The profiling indicated that the culprit is in incrementing the weights for each tag in *scores*. I eventually used an Accumulator as a wrapper around the defaultdict, which yielded an improvement.
Other attempts only resulted in longer, less-readable code. The only remaining option would be again to change the data types of `sp.feature_weights` and/or of *scores*. 

Altogether, I was able to achieve savings of around 30-40% in computational speed compared to the initial stage before profiling. \
I believe there is still room for improvement, due to the aforementioned outstanding issues.


### TO DO
---
- fully optimize the model, addressing the bottlenecks mentioned above
- try and introduce memoization when computing and aggregating features
- accuracy was not my focus in this case, but a natural next step would be to turn this into an averaged perceptron, that would reduce the impact of the randomness in shuffling the samples during training.
- use a validation set to monitor performance during training, since training accuracy is clearly a bad estimate of testing accuracy and encourages overfitting. This can be done by simply adding a `dev_file` to the parameters of `fit!()` and a call to `evaluate()` on it at the end of each iteration. I opted not to do this because I would need another small dataset to use for development, in order not to reduce the training set further, and all of those that I could find were very large and used more complicated sets of POS tags. In general, even when fully optimized, the structured perceptron is not a fast model by all means and using more data would only make this project cumbersome to run, without adding anything interesting in terms of coding. 

The datasets used here as examples are very small and the tag set is limited, thus convergence to almost perfect training accuracy and high testing accuracy (~90%) occurs in about 5-7 epochs, for both inference methods. \
Moreover, in this case, greedy inference, which is also much faster to train and converge, works just as well as dynamic programming.

### References
---
- [Collins, M. (2002)](https://www.aclweb.org/anthology/W02-1001.pdf), "Discriminative Training Methods in Hidden Markov Models: Theory and Experiments with Perceptron Algorithms", Proceedings of the Conference on EMNLP 


