import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as si
import seaborn as sns

sns.set_theme()

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

# Read the merged data.
df = pd.read_pickle("concatenated.pkl")

# Report the class balance of the whole dataset.
value_counts = df["label"].value_counts()

# Plot the frequency distribution by label.
# ax = sns.countplot(df, x="label", order = value_counts.index)
# ax.set_xticks(list(range(0, value_counts.size)))
# new_labels = [label_map.get(int(label.get_text())) for label in ax.get_xticklabels()]
# ax.set_xticklabels(new_labels)
# ax.set_xlabel("Activity")
# ax.set_ylabel("Datapoint Count")
# # Rotate long x axis labels. Same as , rotation=45, ha="right"
# ax.figure.autofmt_xdate()
# plt.tight_layout()
# plt.show()

df = df[df["file"].isin(["S007"])]

# Plot Univariate Analysis of sensor data by X,Y,Z for back, thigh locations
# new_labels = [label_map.get(int(label.get_text())) for label in ax.get_xticklabels()]
features = ['back_x', 'back_y', 'back_z', "thigh_x", "thigh_y", "thigh_z"]
# titles = ['Feature: back_x', 'Feature: back_y', 'Feature: back_z', 'Feature: thigh_x', 'Feature: thigh_y', 'Feature: thigh_z']
fig, axs = plt.subplots(2, 3, figsize=(10, 8))
for i, ax in enumerate(axs.flatten()):
    sns.boxplot(x="label", y=features[i], data=df, showfliers=False, palette="muted", hue="label", legend=False, ax=ax)
    # ax.set_title(titles[i])
    ax.set_xticks(list(range(0, value_counts.size)))
fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.1, wspace=0.3, hspace=0.3)
plt.legend([f"{key}: {value}" for key, value in label_map.items()], 
    loc="best",
    bbox_to_anchor=(0.5, 2.3, 0.5, 0.5),
    ncol=5)
plt.show()