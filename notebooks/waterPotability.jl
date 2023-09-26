# Import libraries
using CSV, DataFrames

# Import some 
using Statistics: mean, std

# Import oversample function for imbalance data
using MLUtils: oversample

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

# Plot predictor distribution (histogram)
plot([histogram(predictors[!, col]; label=col) for col in names(predictors)]...)
savefig("waterPotabilityPredictorsHistogram.png") 