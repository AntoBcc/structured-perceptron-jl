module SP

include("SP_reader.jl") #functions to load and pre-process data

using DataStructures 
using FileIO
using JLD2
using Parameters
using Printf 
using Random
using StatsBase 


"""
The Structured Perceptron is initialized as a struct with default parameters.
It must be mutable, since its parameters are updated during training. 

Parameters:

        1) Trainable:
        ´tags´: set of all possible tags
        ´feature_weights´: nested dictionary of form (feature=>Dict(tag=>weight)) 
                        that maps each feature to weights associated to different tags
        ´tag_dict´: default dictionary of form (word=>Set_of_tags) that maps each word 
                    to the set of tags that can be associated  with it
        
        1) Non-trainable:
        ´start_´: start token
        ´end_´: end token

´´´
The way Base.show has been overloaded makes it possible to distinguish if
we are initialising a perceptron from scratch or loading a trained model.
"""

@with_kw mutable struct StructuredPerceptron

    tags::Set{AbstractString} = Set() 
    feature_weights::Dict{AbstractString,Dict{AbstractString,Float64}} = Dict{AbstractString,Dict{AbstractString,Float64}}()
    tag_dict::DefaultDict{AbstractString,Set} = DefaultDict{AbstractString, Set}(Set{AbstractString})

    start_::AbstractString = "_START_"
    end_::AbstractString = "_END_"

end


function Base.show(io::IO, sp::StructuredPerceptron)
    status = "initialized"
    if length(sp.tags) > 0 
        status = "loaded"
    end
    println(io, "Structured Perceptron $status")
end



function fit!(sp::StructuredPerceptron, file_name::AbstractString, iterations::Int64, prediction_method::AbstractString; 
            learning_rate::Float64=0.25, verbose::Bool=true)
    
    """
    fit!() reads training data from ´file_name´ in ConLL format, 
    computes predictions with the ´prediction_method´ of choice,
    then compares the predicted tags with the true tags and updates 
    the weight vector of the perceptron ´sp´, at the chosen ´learning_rate´ 
    ´´´
    If ´verbose´ = true, it prints updates on progress, number of features and 
    training accuracy for each of the ´iterations´ 
    """

    #read data and create an array of instances, as an array of (words,tags) tuple for each sentence:
    generator = SP_reader.read_conll_file(file_name) 
    instances = []
    for (words, tags) in generator
        push!(instances,(words,tags))
        union!(sp.tags,Set(tags))
        for (word, tag) in zip(words, tags)
            push!(sp.tag_dict[SP_reader.normalize!(word)],tag)
        end
    end
    
    for iter in 1:iterations
        correct = 0 
        total = 0
        if verbose
            println("*"^15)
            println("Iteration $(iter)")
        end
        #for each sentence, compute predictions:
        for (i,instance) in enumerate(instances) 
            if verbose
                if i > 0
                    if i%1000==0 
                        println("$i")
                    elseif i%20==0
                        print(".")
                    end
                end
            end
            prediction = predict(sp,instance[1],prediction_method)
            #extract features associated with the true and predicted tags:
            global_gold_features, global_prediction_features = get_global_features(sp,instance[1],prediction,instance[2])
            #update weight vector:
            #move closer to the true tag:
            for (tag, f_ids) in global_gold_features
                for f_id in f_ids
                    id,count = f_id
                    current_count = get!(get!(sp.feature_weights,id,Dict()),tag,0)
                    sp.feature_weights[id][tag] = current_count + (count * learning_rate)
                end
            end
            #move away from the wrong tag:
            #if predicted tag=true tag, the two steps cancel out
            for (tag, f_ids) in global_prediction_features
                for f_id in f_ids
                    id,count = f_id
                    current_count = get!(get!(sp.feature_weights,id,Dict()),tag,0)
                    sp.feature_weights[id][tag] = current_count - (count * learning_rate)
                end
            end 
            #update training accuracy:
            correct += sum((prediction.==instance[2])) 
            total += length(prediction)
        end      

        if verbose
            println("$(length(sp.feature_weights)) features") 
            println("Training accuracy: $(@sprintf("%.2f",round(correct/total,digits=2)))")
        end
        #shuffle sentences at each iteration:
        shuffle!(instances) 
    end
end
    

function get_features(word::AbstractString, previous_tag2::AbstractString, previous_tag::AbstractString, words::AbstractArray, i::Int64)
    
    """
    get_features() computes features of interest for a given word in a sentence, using:
        - ´word´: the word itself
        - ´previous_tag2´, ´previous_tag´: the two previous tags (2nd order HMM)
        - ´words´: the whole tokenized sentence 
        - ´i´: index to keep track of the position of the word in the sentence
    ´´´
    Features are returned as an array of strings.
    """

    prefix = word[1:min(3,end)] 
    suffix = word[max(1,end-2):end] 

    features = [
        "PREFIX=$prefix",
        "SUFFIX=$suffix",
        "LEN<=3=$(length(word)≤3)", 
        "WORD=$word","NORM_WORD=$(words[i])",
        "PREV_WORD=$(words[i-1])",
        "PREV_WORD_PREFIX=$(words[i-1][1:min(3,end)])",
        "PREV_WORD_SUFFIX=$(words[i-1][max(1,end-2):end])",
        "PREV_WORD+WORD=$(words[i-1])+$(words[i])", 
        "NEXT_WORD=$(words[i+1])",
        "NEXT_WORD_PREFIX=$(words[i+1][1:min(3,end)])",
        "NEXT_WORD_SUFFIX=$(words[i+1][max(1,end-2):end])",
        "WORD+NEXT_WORD=$(words[i])+$(words[i+1])",
        "NEXT_2WORDS=$(words[i+1])+$(words[i+2])",
        "PREV_TAG=$(previous_tag)",
        "PREV_TAG_BIGRAM=$previous_tag2+$previous_tag",
        "PREV_TAG+WORD=$previous_tag+$word",
        "PREV_TAG+SUFFIX=$previous_tag+$suffix",
        "PREV_TAG+PREFIX=$previous_tag+$prefix",
        "WORD+TAG_BIGRAM=$word+$previous_tag2+$previous_tag",
        "SUFFIX+2TAGS=$suffix+$previous_tag2+$previous_tag",
        "PREFIX+2TAGS=$prefix+$previous_tag2+$previous_tag",
        "BIAS"]
    
    return features
end


function get_global_features(sp::StructuredPerceptron, words::AbstractArray, predicted_tags::AbstractArray, true_tags::AbstractArray)
    
    """
    get_global_features() sums up local (= word-wise) features in a tokenized sentence, ´words´,
    aggregating counts over ´predicted_tags´ and ´true_tags´. 
    Start and end tokens from ´sp´ are used for padding.
    ´´´
    It returns two defaultdicts of form (tag=>[(feature1,count1),...,(featureN,countN)])
    """

    context = [sp.start_;SP_reader.normalize!.(words);[sp.end_, sp.end_]]

    global_prediction_features = DefaultDict{AbstractString, Any}(AbstractString[])  
    global_gold_features = DefaultDict{AbstractString, Any}(AbstractString[]) 

    prev_pred_tag = sp.start_
    prev_pred_tag2 = sp.start_

    for (j, word,predicted_tag,true_tag) in zip(eachindex(words), words, predicted_tags, true_tags) 
        prediction_features = SP.get_features(word, prev_pred_tag2, prev_pred_tag, context, j+1)
        global_prediction_features[predicted_tag] = vcat(global_prediction_features[predicted_tag],prediction_features)
        global_gold_features[true_tag] = vcat(global_gold_features[true_tag],prediction_features)

        prev_pred_tag2 = prev_pred_tag
        prev_pred_tag = predicted_tag
    end
    #for each key, create a list of (feature,count) tuples as value
    global_gold_features = Dict(k=>[(feat,count) for (feat,count) in countmap(v)] for (k,v) in global_gold_features)
    global_prediction_features = Dict(k=>[(feat,count) for (feat,count) in countmap(v)] for (k,v) in global_prediction_features)

    return global_gold_features, global_prediction_features
end


function get_scores(sp::StructuredPerceptron, features::AbstractArray) 
    
    """
    get_scores() computes scores associated to each tag, given an array of ´features´ 
    and using current weights from ´sp´.
    ´´´
    Scores are returned as a default dictionary, of form (tag=>score)
    """
    
    scores = Accumulator(DefaultDict{AbstractString,Float64}(0.0))

    for feature in features
        feature ∉ keys(sp.feature_weights) && continue
        weights = sp.feature_weights[feature]
        for (tag, weight) in weights
            inc!(scores, tag, weight) 
        end
    end
    #if there  are no scores, just return the first tag with score 1.0:
    if isempty(scores) 
        tags = collect(sp.tags)
        scores[tags[1]] = 1 
    end

    return scores 
end 


function predict(sp::StructuredPerceptron, words::AbstractArray, prediction_method::AbstractString)
    
    """
    predict() returns tag predictions for a tokenized sentence (´words´), based on the chosen ´prediction_method´
    and using current parameter updates of ´sp´
    ´´´
    it returns predicted tags as an array of strings
    """

    return (prediction_method == "viterbi" ? predict_viterbi(sp,words) : predict_greedy(sp,words))
end


function predict_viterbi(sp::StructuredPerceptron, words::AbstractArray) 
    
    """
    predict_viterbi() performs inference using Viterbi algorithm, a dynamic programming approach.
    ´´´
    It returns predicted tags for a sentence as an array of strings
    """

    context = [sp.start_;SP_reader.normalize!.(words);[sp.end_, sp.end_]]
    N = length(words) #dimension of the observations space
    M = length(sp.tags) #dimension of the state space
    tags = sort!(collect(sp.tags))  

    Q = fill(-Inf,N,M)  #to hold viterbi scores
    backpointers = fill(M,N,M) #pointers to the tag associated with the previous word
    #initialise probabilities for tag j at position 2 (first actual word of the sentence):
    features = get_features(words[1],sp.start_,sp.start_,context,2)
    scores = get_scores(sp,features) 
    allowed_initial_tags = sp.tag_dict[context[2]] 

    @inbounds for j in 1:M 
        #for each tag, if it is an allowed one (or the allowed set is still empty), get the score
        if isempty(allowed_initial_tags) || tags[j] in allowed_initial_tags
            Q[1,j] = scores[tags[j]]
        end
    end
    @inbounds for i in 2:N 
        #for each word, get allowed tags:
        allowed_tags = sp.tag_dict[context[i+1]]  
        #for every possible previous tag:
        @inbounds for j in 1:M
            best_score = 0.0
            prev_tag = tags[j]  
            allowed_previous_tags =  sp.tag_dict[context[i]]
            #skip impossible tags:
            if !isempty(allowed_previous_tags) && !(prev_tag in allowed_previous_tags)
                continue 
            end
            best_before = Q[i-1,j] 
            #for every possible pre-previous tag:
            @inbounds for k in 1:M
                if i == 2 
                    prev2_tag = sp.start_
                else 
                    prev2_tag = tags[k]  
                    allowed_previous2_tags = sp.tag_dict[context[i-1]]
                    #skip impossible tags
                    if !isempty(allowed_previous2_tags) && !(prev2_tag in allowed_previous2_tags)
                        continue
                    end
                end
                # get features and compute scores for current word, given the two previous tags:
                features = get_features(words[i], prev2_tag, prev_tag, context, i+1)
                scores = get_scores(sp,features)
                #update best score:
                @inbounds for t in 1:M
                    tag = tags[t] 
                    #if word unseen, use all tags, else use allowed ones:
                    if isempty(allowed_tags) || tag in allowed_tags
                        tag_score = best_before + scores[tag] 
                        #if all tag_scores are negative, best_score is not updated neither is backpointers
                        if tag_score > best_score
                            Q[i,t] = tag_score
                            best_score = tag_score
                            backpointers[i,t] = j 
                        end
                    end
                end
            end
        end
    end

    best_id = argmax(Q[end,:]) 
    predtags = [tags[best_id]]
    @inbounds for i in N:-1:2
        idx = backpointers[i,best_id] 
        push!(predtags,tags[idx]) 
        best_id = idx 
    end
    
    return reverse(predtags)
end


function predict_greedy(sp::StructuredPerceptron, words::AbstractArray)
    
    """
    predict_greedy() performs inference using a greedy approach.
    ´´´
    It returns predicted tags for a sentence as an array of strings 
    """
    
    context = [sp.start_;SP_reader.normalize!.(words);[sp.end_, sp.end_]]
    prev_predicted_tag = sp.start_
    prev_predicted_tag2 = sp.start_

    result = AbstractString[] 

    for (i, word) in enumerate(words)
        #if the word is unambiguous (just one entry in the tag_dict), just look up the tag:
        predicted_tag = ""
        if length(sp.tag_dict[word]) == 1
            predicted_tag = collect(sp.tag_dict[context[i+1]])[1] 
        end
        #greedy approach, based on tag scores:
        if isempty(predicted_tag)
            prediction_features = get_features(word, prev_predicted_tag2, prev_predicted_tag, context, i+1)
            scores = get_scores(sp, prediction_features)
            predicted_tag = collect(keys(scores))[argmax(collect(values(scores)))]
        end
        prev_predicted_tag2, prev_predicted_tag = prev_predicted_tag, predicted_tag
        push!(result,predicted_tag)    
    end

    return result
end


function evaluate(sp::StructuredPerceptron, test_file::AbstractString, prediction_method::AbstractString)::Float64
    
    """
    evaluate() returns prediction accuracy over a ´test_file´ (in CONLL format), 
    using a (trained) perceptron ´sp´ and the ´prediction_method´ of choice.
    ´´´
    It returns accuracy as a floating point value in [0,1]
    """

    generator = SP_reader.read_conll_file(test_file)

    instances = []
    for (words, tags) in generator
        push!(instances,(words,tags))
    end

    correct = 0
    total = 0

    for instance in instances
        predicted_tags = predict(sp,instance[1],prediction_method)
        correct += sum((predicted_tags.==instance[2])) 
        total += length(predicted_tags)
    end

    return round(correct/total,digits=2)
end



function save_model(sp::StructuredPerceptron, out_name::AbstractString) 

    """
    save_model() saves trained Structured Perceptron object ´sp´ to a jld2 file called ´out_name´
    """

    endswith(out_name,".jld2") || throw(ArgumentError("file name must end in .jld2"))
    
    println("Saving Structured Perceptron to $out_name...")
    save(out_name,"SP",sp)
    println("Done!")
end


function load_model(in_name::AbstractString)::StructuredPerceptron

    """
    load_model() loads a saved Structured Perceptron object from a jld2 file called ´in_name´
    """

    endswith(in_name,".jld2") || throw(ArgumentError("file name must end in .jld2")) 

    println("loading Structured Perceptron from $in_name...")
    return load(in_name)["SP"]
end


end

