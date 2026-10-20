import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
file_path = "2026_01_17/invZ/invZ_summary_los.csv" 
df = pd.read_csv(file_path)

# Filter data to only include K up to 7
df = df[df['K'] <= 7]

# 2. Setup visual style
sns.set_theme(style="whitegrid")

# 3. Create the FacetGrid
g = sns.catplot(
    data=df,
    x="N",
    y="percentage",
    col="K",
    kind="bar",
    col_wrap=3,
    height=3, 
    aspect=1.5,
    palette="viridis",
    sharex=False 
)

# 4. Customize Labels and Titles
# We use set_axis_labels for Y, but we will handle X inside the loop for every subplot
g.set_axis_labels("", "Percentage (%)") 
g.set_titles("Users (K) = {col_name}")

# Add values on top of bars AND set x-labels for every subplot
for ax in g.axes.flat:
    # 1. Force the x-axis title to appear on every subplot
    ax.set_xlabel("Number of antennas (N)", fontsize=10)
    
    # 2. Force the tick labels (numbers) to be visible
    ax.tick_params(labelbottom=True) 
    
    # 3. Add the bar labels (percentages)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=8)

# 5. Adjust Layout and Save
# Note: hspace needs to be higher (e.g., 0.6) to make room for labels between rows
plt.subplots_adjust(top=0.9, hspace=0.6) 
g.fig.suptitle('Percentage of Z-compatible Channels per Antenna Count (N)')

plt.savefig("plot1.png", dpi=300)
plt.show()