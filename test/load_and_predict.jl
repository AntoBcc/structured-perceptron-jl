include("SP.jl")

using WordTokenizers

data_dir = "data/";
test_file = data_dir*"tiny_POS_test.data";
save_dir = "trained_models/";
save_file = save_dir*"trained_sp.jld2";

trained_sp = SP.load_model(save_file)

println("Test-set accuracy: $(SP.evaluate(trained_sp,test_file,prediction_method))")

prediction_method = "greedy";
example = tokenize.("As Wall Street strengthened, the London trading room went wild.");
println("test sentence: $(example)")
println("predicted tags: $(string.(SP.predict(trained_sp,example,prediction_method)))")
