import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
file_path = "2026_01_21/invZ/invZ_summary_noisyMMSE.csv"  # Ensure this matches your csv name
df = pd.read_csv(file_path)

# 2. Setup the visual style
sns.set_theme(style="whitegrid")

# 3. Create the FacetGrid (Small Multiples)
# col="K": Creates a new subplot for each value of K
# col_wrap=3: Wraps to a new line after 3 plots (creating a grid)
# sharex=False: Allows X axis labels to be independent (optional, but good if L varies)
g = sns.catplot(
    data=df,
    x="L",
    y="percentage",
    col="K",
    kind="bar",
    col_wrap=3,
    height=3, 
    aspect=1.5,
    palette="viridis"
)

# 4. Customize Labels and Titles
g.set_axis_labels("Number of APs (L)", "InvZ Percentage (%)")
g.set_titles("Users (K) = {col_name}")

# Optional: Add values on top of bars
for ax in g.axes.flat:
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=8)

# 5. Adjust Layout and Save
plt.subplots_adjust(top=0.9) # Make room for main title
g.fig.suptitle('Percentage of InvZ Matrices per AP Count (L) for each User Count (K)')

plt.savefig("2026_01_21/invZ/invZ_plot_noisyMMSE.png", dpi=300)
plt.show()