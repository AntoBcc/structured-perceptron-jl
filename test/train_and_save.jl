include("../src/SP.jl")

data_dir = "data/";
file = data_dir*"tiny_POS_train.data";
save_dir = "trained_models/";
save_file = save_dir*"trained_sp.jld2";

sp = SP.StructuredPerceptron()
prediction_method = "greedy";
epochs = 5; 
SP.fit!(sp,file,epochs,prediction_method) 

SP.save_model(sp,save_file)

