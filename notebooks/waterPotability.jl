# import libraries
using CSV, DataFrames

# import some 
using Statistics: mean, std

# import oversample function for imbalance data
using MLUtils: oversample

# Import Plots, and StatsPlots for visualizations and diagnostics.
using Plots: plot, plot!, scatter!
using StatsPlots

# =============================
PATH = "/Users/danieldiazalmeida/Downloads/DataScienceDatasets/water_potability/"
# train data file
data = "water_train.csv" 

# read data with CSV and transform it to DataFrame
df = DataFrame(CSV.File(PATH * data))
print(first(df))
println(last(df, 1))

# inspecting DataFrame shape
df_shape = nrow(df), ncol(df)
println("The DataFrame shape is $df_shape")

# describe data
println(describe(df))

# dataset presents zero missing values for all features and target