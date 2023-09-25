println(versioninfo())

# Importing data handling libraries
using CSV, DataFrames, TimeSeries

using Statistics: mean, std
using Impute

# Importing partial autocorrelation function
using StatsBase: pacf, autocor

# Functionality for splitting and normalizing the data
using MLUtils#: shuffleobs, rescale!
using FeatureTransforms: StandardScaling, fit!, apply!

# Import Plots, and StatsPlots for visualizations and diagnostics.
using Plots: plot, plot!, scatter!
using StatsPlots


# -------------------------------
PATH = "/Users/danieldiazalmeida/Downloads/DataScienceDatasets/"
data = "aapl_raw_data.csv"

df = DataFrame(CSV.File(PATH*data))
println(first(df, 1))
println(last(df, 1))

# describing DataFrame
describe(df)

# inspecting DataFrame shape
df_shape = nrow(df), ncol(df)
println("The DataFrame shape is $df_shape")

# ploting the distribution of open, high, low and close predictors
plot([histogram(df[!, col]; label=col) for col in ["open","high","low","close"]]...)

# creating a dates vector 
dates = Date(1980, 12, 12):Day(1):Date(2023, 08, 25)
# creating a temporary DataFrame
tempdf = DataFrame(date = dates)


alldf = outerjoin(tempdf, df, on = :date, order=:left)
dfopen = select(alldf, [:date, :open])
first(dfopen)

# plotting TimeSeries DataFrame
plot(dfopen[!, :date], dfopen[!, :open]; 
    xlabel="date", ylabel="AAPL Open (USD)", legend=false, marker=:o)

# interpolating data to fill missing values
dfopen.open = Impute.interp(dfopen[!, :open])
# plotting TimeSeries with missing values filled 
plot(dfopen[!, :date], dfopen[!, :open]; 
    xlabel="date", ylabel="AAPL Open (USD)", legend=false, marker=:o)


# ==================
# Modeling
# ==================
using Flux, ProgressMeter
using Flux.Losses: mse


df = TimeArray(dfopen, timestamp = :date)
println(first(df))

# data spliting index
split_idx = (nrow(dfopen)*.80) |> round |> Int
# train dates 
traindate = dates[1:split_idx]
# test data selection by date
testdate = dates[split_idx:end]

plot(df[traindate]; label="Train")
plot!(df[testdate]; label="Test")

# splitting data into train and test sets
train_data = df[traindate] |> values
test_data = df[testdate] |> values

train_data = collect(skipmissing(train_data))
test_data = collect(skipmissing(test_data))

# standardizing
temp_scaling = StandardScaling()
fit!(temp_scaling, train_data)
apply!(train_data, temp_scaling)
apply!(test_data, temp_scaling)

# defining functions to get sequences
floater(x) = Float32.(x)

function get_seq(data::Vector{Float64}, step::Int64)
    """
    """
    seq_length = (length(data) - step)
    x = [data[i:(i + step - 1)] for i = 1:(seq_length)]
    y = [data[(i + step)] for i = 1:(seq_length)]

    x = map(floater, x)
    y = map(floater, y)
    return x, y
end

Xtrain, ytrain = get_seq(train_data, 1)
xtest, ytest = get_seq(test_data, 1)

# creating DL model
in_size = 1
hidden_size = 16
out_size = 1

function getmodel(in_size, hidden_size, out_size)
    """

    """
    return Chain(
        LSTM(in_size => hidden_size),
        #Flux.relu,
        Dense(hidden_size => hidden_size, identity),
        NNlib.elu,
        Dense(hidden_size => out_size, identity)
    )
end

model = getmodel(in_size, hidden_size, out_size)

# defining loss function
loss(x, y) = mse(model(x), y)

# ===================
# Model training
# ===================
η = Float32(.001)
opt = Adam(η)

function traninsetp(opt, model, Xtrain, ytrain; EPOCHS=100)
    opt_state = Flux.setup(opt, model)
    # option 3
    for i=1:EPOCHS
        for d in zip(Xtrain, ytrain)
            grads = Flux.gradient(model) do m
                loss(d...)
            end
        Flux.update!(opt_state, model, grads[1])
        end
    end

end

traninsetp(opt, model, Xtrain, ytrain; EPOCHS=100)

# Results visualization
pred = model(xtest[end])
loss(xtest[end], ytest[end])
plot(2800:length(ytest), ytest[2800:end])
scatter!([length(ytest)], [pred], ms=3.)

# ===================
# Stationary Approach
# ===================
diffdatatrain = diff(train_data)
diffdatatest = diff(test_data)
plot(traindate[1:(end-1)], diffdata; label="Stationary TS")

Xtrain2, ytrain2 = get_seq(diffdatatrain, 1)
xtest2, ytest2 = get_seq(diffdatatest, 1)

model = getmodel(1, 8, 1)

# train model
traninsetp(opt, model, Xtrain2, ytrain2; EPOCHS=1_000)

pred = model(xtest[end-1])[1]
loss(xtest[end-1], ytest[end-1])

rs_ytest = ytest2[2800:end-1] + test_data[2802:end-1] #|> plot
plot(2800:(length(ytest2)-1), rs_ytest)
scatter!([length(ytest2)-1], [pred + test_data[end-1]], ms=3.)