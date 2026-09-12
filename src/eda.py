import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_missing_values(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    return plt.gcf()

def plot_target_distribution(df, target, task_type='classification'):
    plt.figure(figsize=(8, 5))
    if task_type == 'classification':
        sns.countplot(x=target, data=df, palette='Set2')
        plt.title(f'Distribution of Target Variable: {target}')
    else:
        sns.histplot(df[target], kde=True, color='blue', bins=30)
        plt.title(f'Distribution of Target Variable: {target}')
    plt.tight_layout()
    return plt.gcf()

def plot_univariate_numeric(df, col):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Univariate Distribution: {col}')
    plt.tight_layout()
    return plt.gcf()

def plot_univariate_categorical(df, col):
    plt.figure(figsize=(10, 5))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index, palette='pastel')
    plt.title(f'Univariate Distribution: {col}')
    plt.tight_layout()
    return plt.gcf()

def plot_boxplots(df, cols):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[cols], orient='h', palette='Set3')
    plt.title('Boxplots for Numeric Features')
    plt.tight_layout()
    return plt.gcf()

def plot_correlation_heatmap(df, cols):
    plt.figure(figsize=(10, 8))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    return plt.gcf()

def plot_pairplot(df, cols, hue=None):
    if hue:
        g = sns.pairplot(df[cols + [hue]], hue=hue, palette='husl')
    else:
        g = sns.pairplot(df[cols])
    return g.fig

def plot_bivariate_scatter(df, x_col, y_col, hue=None):
    plt.figure(figsize=(8, 6))
    if hue:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, palette='viridis')
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.title(f'Scatter Plot: {x_col} vs {y_col}')
    plt.tight_layout()
    return plt.gcf()

def plot_bivariate_box(df, cat_col, num_col):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=cat_col, y=num_col, palette='Set2')
    plt.title(f'Box Plot: {cat_col} vs {num_col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()
