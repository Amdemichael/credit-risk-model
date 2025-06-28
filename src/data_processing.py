# src/data_processing.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def convert_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df['transactionstarttime'] = pd.to_datetime(df['transactionstarttime'], errors='coerce')
    return df


def get_summary_statistics(df: pd.DataFrame):
    print("🔍 Summary Statistics:\n", df.describe())
    print("\n🔍 Missing Values:\n", df.isnull().sum())
    print("\n🔍 Data Types:\n", df.dtypes)


def plot_numerical_distributions(df: pd.DataFrame, num_cols: list):
    for col in num_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=col, bins=50, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()


def plot_categorical_distributions(df: pd.DataFrame, cat_cols: list):
    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.xlabel("Count")
        plt.ylabel(col)
        plt.grid(True)
        plt.show()


def correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
