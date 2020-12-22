include("../src/SP.jl")

using WordTokenizers 

data_dir = "data/";
file = data_dir*"tiny_POS_train.data";
test_file = data_dir*"tiny_POS_test.data";

### TRAINING THE STRUCTURED PERCEPTRON

#setting some parameters
prediction_method = "viterbi" 
epochs = 5 #N.B: to get full convergence, a couple more iterations might be needed

#initializing and fitting the perceptron
sp = SP.StructuredPerceptron()
@time SP.fit!(sp,file,epochs,prediction_method) 


### PREDICTION AND EVALUATION 

#computing test accuracy
SP.evaluate(sp,test_file,prediction_method) 

"""
#to visualise what the perceptron has learnt so far, you can make calls to its parameters. For instance:

sp.tags #returns all of the possible tags 
sp.tag_dict["attack"] #returns the possible tags associated with the word "attack"
sp.feature_weights["PREV_TAG=VERB"] #returns weights given to different tags, given that the word of interest is preceded by a verb
"""

#some sample sentences for prediction (from the test set)
examples = ["This is panic buying!","As Wall Street strengthened, the London trading room went wild."];

#you can feed any sentence to the predict function as a string, as long as you tokenize it first!
example = tokenize.(examples[1]);
SP.predict(sp,example,prediction_method)


### SAVING AND LOADING MODELS

save_dir = "trained_models/"
save_file = save_dir*"viterbi_sp.jld2" 

#saving the trained model (this will several seconds)
#SP.save_model(sp,save_file)

#loading the saved model (this will several seconds)
#trained_sp = SP.load_model(save_file)

#prediction_method = "viterbi"
#SP.predict(trained_sp,example,prediction_method)


