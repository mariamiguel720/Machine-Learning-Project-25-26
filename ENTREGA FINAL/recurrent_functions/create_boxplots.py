import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil


def create_boxplots(df, numeric_cols, n_cols=2, figsize=(20, 12)):
    """
    Creates boxplots for numeric columns with consistent formatting.
    
    Parameters:
        df (DataFrame): The dataset
        numeric_cols (list): List of numeric columns to plot.
        n_cols (int): Number of subplot columns (default=2)
        figsize (tuple): Base figure size; height is scaled dynamically
    """

    n_features = len(numeric_cols)
    n_rows = ceil(n_features / n_cols)

    # Dynamic height scaling
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(figsize[0], figsize[1])
    )
    axes = axes.flatten()

    sns.set_theme(style="whitegrid", palette="pastel")

    for i, col in enumerate(numeric_cols):
        ax = axes[i]

        sns.boxplot(
            x=df[col],
            ax=ax,
            color='skyblue',
            medianprops={"color": "darkblue", "linewidth": 2},
            boxprops={"alpha": 0.7}
        )

        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.grid(True, linestyle="--", alpha=0.4)

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Boxplots of Numeric Features", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
