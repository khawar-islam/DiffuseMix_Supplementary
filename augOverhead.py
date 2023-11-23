import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib

#matplotlib.rcParams['font.size'] = 14  # Adjust the font size for the axis ticks


# Load your data
data = pd.read_excel('/home/cvpr/Documents/augmentationOverhead.xlsx')  # Replace 'your_file_path.xlsx' with the actual path to your Excel file

# Set the style for the plot
sns.set(style="whitegrid")

# Creating the scatter plot with labels for each method
plt.figure(figsize=(7, 5))
# Define your colors and markers
colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta']  # Adjust as needed
markers = ['o', 's', '^', 'P', '*', 'X', 'D']  # Adjust as needed


plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'

# Iterate through the methods to plot each one with a label
for i, method in enumerate(data['Methods'].unique()):
    color = colors[i % len(colors)]  # Cycle through colors
    marker = markers[i % len(markers)]  # Cycle through markers
    subset = data[data['Methods'] == method]
    plt.scatter(subset["Augmentation Overhead (+%)"], subset["Accuracy (%)"], s=100, marker=marker, color=color)
    for x, y in zip(subset["Augmentation Overhead (+%)"], subset["Accuracy (%)"]):
        plt.text(x + 0.2, y+0.3, method, fontsize=11, ha='center', va='bottom')

# Adding titles and labels
#plt.title('Accuracy vs Augmentation Overhead for Different Methods', fontsize=16)
plt.ylim(top=80)  # You can also set the bottom limit if needed, e.g., plt.ylim(bottom=60, top=80)
plt.xlim(right=400)  # You can also set the left limit if needed, e.g., plt.xlim(left=0, right=400)

plt.xticks(fontsize=14)  # Adjust font size as needed
plt.yticks(fontsize=14)  # Adjust font size as needed

plt.xlabel('Augmentation Overhead (+%)', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)

# Show the plot
plt.tight_layout()
pdf_file_path = 'augOver.pdf'
plt.savefig(pdf_file_path, format='pdf')
plt.show()
