import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.legend import Legend

#matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 12  # Adjust the font size for the axis ticks

# Load the CSV file from the provided path
file_path = '/home/cvpr/Downloads/val_loss.csv'
data = pd.read_csv(file_path)

# Extracting and restricting the 'Step' column to 100
data = data[data['Step'] <= 60]
steps = data['Step']

# # Specifying the column names for 'Top-1 Accuracy' metrics
# accuracy_columns = [
#     'Res50 - Valid/Top-5 Accuracy',
#     'Res50+Gen - Valid/Top-5 Accuracy',
#     'Res50+Gen+Con - Valid/Top-5 Accuracy',
#     'Res50+DiffuseMix - Valid/Top-5 Accuracy',
# ]

# Specifying the column names for 'Top-1 Accuracy' metrics
accuracy_columns = [
    'Res50 - Valid/Loss',
    'Res50+Gen - Valid/Loss',
    'Res50+Gen+Con - Valid/Loss',
    'Res50+DiffuseMix - Valid/Loss',
]

# Specifying the legend labels for each model
legend_labels = [
    'Res50',
    'Res50 + Gen',
    'Res50 + Gen + Con',
    'Res50 + DiffuseMix',
]

# Setting different colors and styles for each line
colors = ['c', 'g', 'r', 'b']
line_styles = [':', '--', '-.', '-']
extended_line_styles = line_styles * (len(accuracy_columns) // len(line_styles) + 1)

# Plotting the graph
plt.figure(figsize=(6, 4))

for i, col in enumerate(accuracy_columns):
    plt.plot(steps, data[col], label=legend_labels[i], color=colors[i % len(colors)], linestyle=extended_line_styles[i], linewidth=1.5)

# Increasing the text size of x and y axis labels
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Validation Error (%)', fontsize=12)

lines = []  # To store line objects for custom legend
for i, col in enumerate(accuracy_columns):
    line, = plt.plot(steps, data[col], label='_nolegend_', color=colors[i % len(colors)], linestyle=extended_line_styles[i], linewidth=1.5)
    lines.append(line)

leg = Legend(plt.gca(), [lines[0], lines[1], lines[2], lines[3]], ['Res50', 'Res50+Gen', 'Res50+Gen+Con', 'Res50+DiffuseMix'],
             loc='best')
plt.gca().add_artist(leg)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.legend(title='', loc='best')
plt.grid(True)
# Save the plot as a PDF file
pdf_file_path = 'val_loss.pdf'
plt.tight_layout()
plt.savefig(pdf_file_path, format='pdf')
plt.show()
plt.close()


