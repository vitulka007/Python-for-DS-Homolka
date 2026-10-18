import numpy as np

#1
data=np.random.normal(loc=50, scale=15, size=(100, 2))

print("Data shape:", data.shape)
print("Data type:", data.dtype)
print("first 5 rows of data:\n", data[:5])


data[(data < 20)|(data > 80)] = np.nan
print(np.isnan(data).sum())

means = np.nanmean(data, axis=0)


nan_mask = np.isnan(data)
data = np.where(nan_mask, means, data)

data_standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

percentile_high,percentile_low = np.percentile(data_standardized, [75, 25], axis=0)

high_mask = data_standardized > percentile_high
low_mask = data_standardized < percentile_low

print("Number of values above 75th percentile:", high_mask.sum())

data_standardized = np.where(low_mask, percentile_low, data_standardized)

print("Mean: ", np.mean(data_standardized, axis=0))
print("Median: ", np.median(data_standardized, axis=0))
print("Variance: ", np.var(data_standardized, axis=0))
print("Standard Deviation: ", np.std(data_standardized, axis=0))

print("Overall Mean: ", np.mean(data_standardized))

print("Min: ", np.min(data_standardized, axis=0))
print("Max: ", np.max(data_standardized, axis=0))