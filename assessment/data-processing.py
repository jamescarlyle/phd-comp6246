import pandas as pd
import os

# Set up a map to make labels human-readable.
label_map = {
  1: "walking",
  2: "running",
  3: "shuffling",
  4: "stairs (ascending)",
  5: "stairs (descending)",
  6: "standing",
  7: "sitting",
  8: "lying",
  9: "stairs",
  13: "cycling (sit)",
  14: "cycling (stand)",
  130: "cycling (sit, inactive)",
  140: "cycling (stand, inactive)"
}

# Generate a pandas dataframe for a given filename.
# Parse timestamp column as a datetime type.
def read_file(name):
    df = pd.read_csv(name, index_col=["timestamp"], parse_dates=["timestamp"])
    df["file"] = os.path.splitext(os.path.basename(name))[0]
    return df

filepaths = ["./MLT-CW-Dataset/"+f for f in os.listdir("./MLT-CW-Dataset") if f.endswith("007.csv")]
df = pd.concat(map(read_file, filepaths))
df.drop(labels = ["index", "Unnamed: 0"], axis = "columns", inplace = True)

# Sanity check for data loaded.
print(df.head(5))
print(f"Index name: {df.index.name}, index datatype: {df.index.dtype}")
print(f"Columns: {df.columns}")

# Checkpoint concatenated data.
df.to_pickle("concatenated.pkl")

# Write a function to drop data labelled with any type of cycling, the company
# is not currently interested in them. Apply it to the dataset and report the new
# dataset size.
filtered_df = df[~df["label"].isin([10, 13, 14, 130, 140])]
# Note that data with the label 10 has also been filtered out, since label 10 doesn"t correspond with any known activity.

# Write a function to merge the stairs (ascending) and stairs (descending) labels
# in a single class “stairs” with code 9, as the company wants to consider them
# as a single action. Apply it to the dataset
merged_df = filtered_df.replace(to_replace={"label": {4:9, 5:9}}, inplace=False)

merged_df.to_pickle("merged.pkl")

# Use appropriate sampling and visualisations to report and interpret data patterns of the “Walking” class in the dataset.
# Note: use filtered_df as it still contains all walking data, and is smaller.
walking_df = merged_df[filtered_df["label"].isin([1])]
walking_df.index.name = "timestamp"

walking_df.to_pickle("walking.pkl")

# Sample a random subset of the walking data to plot. iloc samples every n rows. head takes the top n rows. 
# Since we want to plot walking data, we need to sample e.g. at a minimum of 10Hz to observe walking movement, rather than sample across hours of walking.
sampled_df = walking_df.iloc[::2].head(1000)

sampled_df.to_pickle("sampled.pkl")
