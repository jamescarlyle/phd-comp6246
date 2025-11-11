import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as si
import seaborn as sns

sns.set_theme()

sampled_df = pd.read_pickle("sampled.pkl")


# new_labels = [label_map.get(int(label.get_text())) for label in ax.get_xticklabels()]
features = ['back_x', 'back_y', 'back_z', "thigh_x", "thigh_y", "thigh_z"]

# Use built-in Savitzky-Golay filter for preserving shape while denoising the data.
fig, axs = plt.subplots(6,1, figsize=(10, 8))
for i, ax in enumerate(axs.flatten()):
  sns.lineplot(sampled_df, x=sampled_df.index, y=sampled_df[features[i]], linewidth=0.2, alpha=0.8, label="Raw Data", ax=ax)
  # sns.lineplot(sampled_df, x=sampled_df.index, y=si.savgol_filter(sampled_df[features[i]], window_length=10, polyorder=3), label="Savitzky-Golay Filter", ax=ax, color='r')
  # NB x axis labels have to be cleared at the plot level, not the figure level.
  if i == 5:
    ax.set_xticks([0,999])
    ax.legend(loc="upper right")
  else:
     ax.set_xticks([])
     ax.set(xlabel=None)
     ax.get_legend().remove()

plt.tight_layout()
plt.show()