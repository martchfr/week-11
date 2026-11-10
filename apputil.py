import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

# Define kmeans function and return the cluster centers and labels
def kmeans(X, k):
    km = KMeans(n_clusters = k, n_init = 'auto')
    km.fit(X)
    return km.cluster_centers_, km.labels_

# Load Diamonds Dataframe from Seaborn Globally
df_diamonds = sns.load_dataset("diamonds")

# Global Variable to hold all Numerical Columns from Diamonds Dataframe
nbr_columns = ["carat", "depth", "table", "price", "x", "y", "z"]

# Create a function that runs kmeans on a specified number of rows from the diamonds dataset (n) and the specified number of clusters (k)
def kmeans_diamonds(n, k):

    df = df_diamonds[nbr_columns][:n]
    
    centroids, labels = kmeans(df, k)

    return centroids, labels