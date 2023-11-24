import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/home/cvpr/Documents/augmentationOverhead.xlsx'
data = pd.read_excel(file_path)

# Extract unique methods
unique_methods = data['Methods'].unique()

# Define colors and markers
colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta']
markers = ['o', 's', '^', 'P', '*', 'X', 'D']

plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'

# Extend colors and markers to ensure enough unique ones for each method
extended_colors = colors * -(-len(unique_methods) // len(colors))  # Ceiling division
unique_markers = markers * -(-len(unique_methods) // len(markers))  # Ceiling division

# Assign unique markers and colors to specific methods
# DiffuseMix - Brown, Diamond Marker
# SaliencyMix - Purple, Plus Marker
# Guided-R - Pink, Hexagon Marker
# Mixup - Yellow, Triangle Down Marker
diffuseMix_index = unique_methods.tolist().index('DiffuseMix')
saliencyMix_index = unique_methods.tolist().index('SaliencyMix')
guided_r_index = unique_methods.tolist().index('Guided-R')
mixup_index = unique_methods.tolist().index('Mixup')

extended_colors[diffuseMix_index] = 'brown'
unique_markers[diffuseMix_index] = 'd'
extended_colors[saliencyMix_index] = 'purple'
extended_colors[guided_r_index] = 'pink'
unique_markers[guided_r_index] = 'h'
extended_colors[mixup_index] = 'yellow'
unique_markers[mixup_index] = 'v'

# Creating the scatter plot
plt.figure(figsize=(7, 5))
sns.set(style="whitegrid")
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'

for i, method in enumerate(unique_methods):
    color = extended_colors[i]
    marker = unique_markers[i]
    subset = data[data['Methods'] == method]
    plt.scatter(subset["Augmentation Overhead (+%)"], subset["Accuracy (%)"], s=100, marker=marker, color=color)
    for x, y in zip(subset["Augmentation Overhead (+%)"], subset["Accuracy (%)"]):
        plt.text(x + 0.2, y + 0.3, method, fontsize=11, ha='center', va='bottom')

# Setting axis limits and labels
plt.ylim(top=80)
plt.xlim(right=400)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Augmentation Overhead (+%)', fontsize=15)
plt.ylabel('Accuracy (%)', fontsize=15)

# Show and save the plot
plt.tight_layout()
pdf_file_path_final = 'augOver_final.pdf'
plt.savefig(pdf_file_path_final, format='pdf')
plt.show()
