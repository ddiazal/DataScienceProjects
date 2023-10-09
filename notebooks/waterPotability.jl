# Import libraries
using CSV, DataFrames

# Import some 
using Statistics: mean, std, cor

# Import oversample function for imbalance data
using MLUtils: oversample, splitobs, getobs, numobs

# Import Plots, and StatsPlots for visualizations and diagnostics.
using Plots: plot, plot!, scatter!
using StatsPlots

# =============================
PATH = "/Users/danieldiazalmeida/Downloads/DataScienceDatasets/water_potability/"
# Train data file
data = "water_train.csv" 

# Read data with CSV and transform it to DataFrame
df = DataFrame(CSV.File(PATH * data))
print(first(df))
println(last(df, 1))

# Inspecting DataFrame shape
df_shape = nrow(df), ncol(df)
println("The DataFrame shape is $df_shape")

# Describe data
println(describe(df))

# Dataset presents zero missing values for all features and target.
# In addition, all predictors range between 0 and 1.

# Lowercase DataFrame column names
rename!(df, strip.(lowercase.(names(df))))
println(names(df))

# Separate target from predictors
target = df[!, :potability]
predictors = select(df, Not(:potability))


# =================================
# Exploring Predictors Data
# =================================
cols = names(predictors)

# Plot predictor distribution (histogram)
plot([histogram(predictors[!, col]; label=col) for col in cols]...)
savefig("waterPotabilityPredictorsHistogram.png") 

plot([boxplot(predictors[!, col]; label=col) for col in cols]...)
savefig("waterPotabilityPredictorsBoxplot.png") 

# correlation
@df predictors corrplot(cols(1:9), grid = false; size=(900,600))

corr = [cor(target, predictors[!, col]) for col in cols]

# heatmap plot for predictors correlation matrix
heatmap(cor(predictors|> Matrix),
    xticks=(1:9, cols),
    yticks=(1:9, cols);
    # plot size
    size=(900,600),
    # rotate x ticks
    xrot=90
)


# =================================
# Data Preprocessing
# =================================
using FeatureTransforms: StandardScaling, fit!, apply!
# determine number of positive and negative samples
npos = sum(target .== 1)
nneg = sum(target .== 0)

# sample observations percentage
pos_obs = npos / df_shape[1]
neg_obs = nneg / df_shape[1]

# positive sample weight
pos_weight = nneg / npos
println("The weight of postive observations is $pos_weight")

# split data into training and testing sets
train, test = splitobs((predictors |> values, target), at=0.8, shuffle=true)
# get the number of observations in each dataset
trainobs, testobs = numobs(train), numobs(test)

# retreive xtrain, ytrain sets 
Xtrain, Ytrain = getobs(train, 1:trainobs)
# retreive xtest and ytest sets 
xtest, ytest = getobs(test, 1:testobs)

# standardize train and test sets
scaler = StandardScaling()
# fit scaler onto train data
fit!(scaler, Xtrain)
# apply fitted scaler to standardize train predictors
apply!(Xtrain, scaler)
# apply fitted scaler to standardize test predictors
apply!(xtest, scaler)

#norm_predictors = mapcols(x -> (x .- mean(x))./std(x), predictors)

mapcols!(x->Float32.(x), Xtrain)
Ytrain = Int32.(Ytrain)
mapcols!(x->Float32.(x), xtest)
ytest = Int32.(ytest)


# =================================
# Modeling
# =================================
using Flux, ProgressMeter
using Flux.Data: DataLoader
using Flux.Losses: logitbinarycrossentropy

# data
Xtrain = Xtrain |> permutedims |> Matrix
Ytrain = reshape(Ytrain, (1, trainobs))
xtest = xtest |> permutedims |> Matrix
ytest = reshape(ytest, (1, testobs))

trainloader = DataLoader((Xtrain, Ytrain), batchsize=32, shuffle=true)
testloader = DataLoader((xtest, ytest), batchsize=32, shuffle=false)

in_features = length(cols)
# define model
model = Chain(
    Dense(in_features => 32, NNlib.elu),
    Dense(32 => 32,  NNlib.elu),
    Dense(32 => 1, identity)
)

# define loss function with positive class weight
pos_weight = Float32(pos_weight)
loss(ŷ, y) = logitbinarycrossentropy(ŷ, y, agg=x->mean(pos_weight .* x))

# optimizer
η = Float32(.001)
opt = Adam(η)


function model_acc(model, testloader)
    acc = []
    Flux.testmode!(model)
    for (xbatch, ybatch) in testloader
        output = model(xbatch)
        ŷ = output .≥ .5
        pred = sum(ŷ .== ybatch) / length(ybatch)
        push!(acc, pred) 
    end
    println("model avg accuracy: $(mean(acc))")
end


function train_step(model, trainloader, EPOCHS)
    # train loss vector
    trainloss_vec = []

    opt_state = Flux.setup(opt, model)
    Flux.trainmode!(model)
    for i=1:EPOCHS
        trainloss = 0
        for (xbatch, ybatch) in trainloader
            grads = Flux.gradient(model) do m
                ŷ = m(xbatch)
                trainloss += loss(ŷ, ybatch)
            end
            Flux.update!(opt_state, model, grads[1])
        end

        if i%2 == 0
            push!(trainloss_vec, trainloss/length(trainloader))
        end

    end
    println("The avg training loss is $(mean(trainloss_vec))")
    println("Done!")

end


train_step(model, trainloader, 2_000)
model_acc(model, testloader)
for epoch = 1:20
    train_step(model, trainloader)
    model_acc(model, testloader)
end


# ==========================
# Training with FluxTraining
# ==========================
using FluxTraining

lossfn = Flux.Losses.logitbinarycrossentropy
optimizer = Flux.ADAM()
learner = Learner(model, lossfn; callbacks=[ToGPU(), Metrics(accuracy)], optimizer)

FluxTraining.fit!(learner, 10, (trainloader, testloader))


# ----------------------
# Training with GPU
using Metal
Metal.versioninfo()

Metal.functional()
device = Flux.get_device(; verbose=true)

model_gpu = device(model) 

# Extract weights and a helper function to reconstruct NN from weights
parameters_initial, reconstruct = Flux.destructure(model_gpu)

ps = Flux.params(model_gpu) |> device
opt_state = Flux.setup(opt, model_gpu) |> device
for i=1:1
    for d in trainloader
        x, y = device(d)
        grads = Flux.gradient(ps) do 
            loss(model_gpu(x), y)
        end
        Flux.Optimise.update!(opt, ps, grads)
    end
end


x, y = first(trainloader)
x = device(x)
y = device(y)

model_gpu(x)
loss(model_gpu(x), y)


opt= Adam(0.001f0)
opt_state = Flux.setup(opt, model_gpu)
for i=1:1
    for d in trainloader
        x, y = device(d)
        Flux.train!(loss, ps, [(x, y)], opt)
        #Flux.train!(loss, model_gpu, [(x, y)], opt_state)
        #Flux.train!((m,x,y) -> loss(x, y), model_gpu, [(x, y)], opt_state)
    end
end

epochs = 10

for epoch = 1:epochs
  for d in trainloader
    gs = gradient(Flux.params(model_gpu)) do
      l = loss(d...)
    end
    update!(opt, Flux.params(model_gpu), gs)
  end
  #@show accuracy(valX, valY)
end



