import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np

# Generic plotting functions
def plot_line_graph(data_x, data_y, title="Line Graph", x_label="X-axis", y_label="Y-axis"):
    plt.figure(figsize=(10, 6))
    plt.plot(data_x, data_y, marker='o')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_bar_graph(df_column, title="Bar Graph", x_label="Categories", y_label="Count"):
    """
    Generic function to plot a bar graph from a DataFrame column using Matplotlib.
    
    Parameters:
    - df_column: pandas Series (e.g., df['column']) containing categorical or binned numerical data
    - title: Title of the bar graph
    - x_label: Label for x-axis
    - y_label: Label for y-axis
    """
    # Calculate frequency of unique values in the column
    value_counts = df_column.value_counts(sort=True).sort_index()
    labels = value_counts.index.astype(str)  # Convert index to string for labels
    values = value_counts.values

    # Create bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='#4B8BBE', edgecolor='#30638E')  # Distinct colors for visibility
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha='right')  # Rotate labels for readability
    plt.tight_layout()
    
    # Display the plot
    plt.show()

def plot_pie_chart(labels, sizes, title="Pie Chart"):
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def plot_scatter_plot(x_data, y_data, title="Scatter Plot", x_label="X-axis", y_label="Y-axis"):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, alpha=0.5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_box_plot(data, title="Box Plot", y_label="Values"):
    plt.figure(figsize=(10, 6))
    plt.boxplot(data)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xticks(range(1, len(data.columns) + 1), data.columns, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_heatmap(data, title="Heatmap", x_labels=None, y_labels=None):
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='YlOrRd', interpolation='nearest')
    plt.title(title)
    plt.colorbar(label='Frequency')
    plt.xticks(range(len(x_labels or data.columns)), x_labels or data.columns, rotation=45, ha='right')
    plt.yticks(range(len(y_labels or data.index)), y_labels or data.index)
    plt.tight_layout()
    plt.show()

def plot_histogram(series, bins=30, title="Histogram", x_label="Values", y_label="Frequency", kde=False):
    data = series.dropna()

    plt.figure(figsize=(10, 6))
    # Histogram
    plt.hist(data, bins=bins, edgecolor="black", alpha=0.6, density=kde)

    # KDE curve
    if kde and len(data) > 1:
        kde_est = gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 200)
        y_vals = kde_est(x_vals)
        plt.plot(x_vals, y_vals, color="red", linewidth=2, label="KDE")
        plt.legend()

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label if not kde else "Density")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_grouped_box_plot(df, group_col, value_col, title="Grouped Box Plot", y_label="Values"):
    plt.figure(figsize=(10, 6))
    groups = [df[value_col][df[group_col] == cat].dropna() for cat in df[group_col].unique()]
    plt.boxplot(groups, labels=df[group_col].unique())
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(group_col)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Example usage (replace with your DataFrame and column)
if __name__ == "__main__":
    # Sample DataFrame for testing (replace with your actual df)
    data = {'column': ['A', 'B', 'A', 'C', 'B', 'A', 'D']}
    df = pd.DataFrame(data)
    
    # Call the function with the desired column
    plot_bar_graph(df['column'], title="Frequency of Categories", x_label="Category", y_label="Count")