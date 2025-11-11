import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

# Read data and assert datatype for vscode intellisense.
all_data = pd.read_pickle("merged.pkl")
assert isinstance(all_data, pd.DataFrame), "Object is not of the expected type DataFrame"

# Split into two dataframes.
# non_s007 = all_data["file"].isin("S007")

sensor_id = "S007"

focus_sensor = all_data[all_data["file"] == sensor_id]
other_sensor = all_data[all_data["file"] != sensor_id]

features = ['back_x', 'back_y', 'back_z', "thigh_x", "thigh_y", "thigh_z"]
for feature in features:
    print(f"Non-null count of sensor {sensor_id} for {feature}: {focus_sensor[feature].count():.2f}, other sensors:{other_sensor[feature].count():.2f}")
    print(f"Mean of sensor {sensor_id} for {feature}: {focus_sensor[feature].mean():.2f}, other sensors:{other_sensor[feature].mean():.2f}")
    print(f"Variance of sensor {sensor_id} for {feature}: {focus_sensor[feature].var():.2f}, other sensors:{other_sensor[feature].var():.2f}")
    print(f"Maximum of sensor {sensor_id} for {feature}: {focus_sensor[feature].max():.2f}, other sensors:{other_sensor[feature].max():.2f}")
    print(f"Minimum of sensor {sensor_id} for {feature}: {focus_sensor[feature].min():.2f}, other sensors:{other_sensor[feature].min():.2f}")
    print("-------")


# When performing a t-test on time series data from two sensors, where one sensor is suspected faulty, it is generally 
# safer not to assume that the variances are equal. Perform Welch’s t-test, which does not assume equal population variance.
# res = ttest_ind(sample_x, other_x, equal_var=False)
features = ['back_x', 'back_y', 'back_z', "thigh_x", "thigh_y", "thigh_z"]

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(f"Distributions of Sensor Data for Sensor {sensor_id}")
for i, ax in enumerate(axs.flatten()):
    ax.set_ylim(0, 100)
    sns.histplot(data=focus_sensor[features[i]], log_scale=False, bins=10, stat="percent", color='red', alpha=0.5, label="sensor", ax=ax)
    hist = sns.histplot(data=other_sensor[features[i]], log_scale=False, bins=10, stat="percent", color='blue', alpha=0.5, label="others", ax=ax)
    ax.scatter(x=focus_sensor[features[i]].min(), y=2, color="red")
    ax.scatter(x=focus_sensor[features[i]].max(), y=2, color="red", label="sensor")
    ax.scatter(x=other_sensor[features[i]].min(), y=2, color="blue")
    ax.scatter(x=other_sensor[features[i]].max(), y=2, color="blue", label="others")
# handles1, labels1 = hist.get_legend_handles_labels()
hist.legend()
plt.tight_layout()
plt.show()