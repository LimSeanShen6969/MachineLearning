import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the preset datasets
df = pd.read_csv('df.csv')
df_cleaned = pd.read_csv('df_cleaned.csv')
df_scaled = pd.read_csv('df_scaled.csv')

# Apply PCA on the scaled data
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

# Streamlit app title
st.title("K-means Clustering with PCA Visualization")

# Sidebar for K-means parameters
st.sidebar.header("K-means Parameters")
n_clusters = st.sidebar.slider("Number of clusters (n_clusters)", min_value=2, max_value=10, value=3, step=1)
init_method = st.sidebar.selectbox("Initialization method (init)", ["k-means++", "random"])
n_init = st.sidebar.slider("Number of initializations (n_init)", min_value=10, max_value=50, value=29, step=1)
max_iter = st.sidebar.slider("Maximum iterations (max_iter)", min_value=100, max_value=500, value=382, step=1)
tol = st.sidebar.slider("Tolerance (tol)", min_value=1e-6, max_value=1e-2, value=1.9266922671867127e-05, format="%.8f", step=1e-6)
algorithm = st.sidebar.selectbox("Algorithm", ["lloyd", "elkan"])

# Run K-means clustering with selected parameters
kmeans = KMeans(
    n_clusters=n_clusters,
    init=init_method,
    n_init=n_init,
    max_iter=max_iter,
    tol=tol,
    algorithm=algorithm,
    random_state=42
)
df_pca['KMeans_Labels'] = kmeans.fit_predict(df_pca[['PC1', 'PC2']]) + 1

# Visualization of clusters
st.subheader("K-means Clustering on PCA Results")
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='KMeans_Labels', data=df_pca, palette='viridis')
plt.title('K-means Clustering On FIFA 19')
st.pyplot(plt)

# Add the cluster labels to the cleaned DataFrame
df_clustered = df_cleaned.copy()
df_clustered['Cluster_Kmeans'] = kmeans.labels_ + 1

# Display the number of records in each cluster
st.subheader("Number of Records in Each Cluster")
cluster_counts = df_clustered['Cluster_Kmeans'].value_counts()
st.write(cluster_counts)

# Display mean statistics for each cluster
st.subheader("Mean Statistics for Each Cluster")
summary = df_clustered.groupby('Cluster_Kmeans').mean()
st.write(summary)

# Display median statistics for each cluster
st.subheader("Median Statistics for Each Cluster")
median_summary = df_clustered.groupby('Cluster_Kmeans').median()
st.write(median_summary)

# Box plots for key features by clusters
st.subheader("Box Plots of Key Features by Cluster")
key_features = df_clustered.columns.drop('Cluster_Kmeans')
for feature in key_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster_Kmeans', y=feature, data=df_clustered)
    plt.title(f'Distribution of {feature} by Cluster')
    st.pyplot(plt)
