import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Load the preset datasets
df = pd.read_csv('df.csv')
df_cleaned = pd.read_csv('df_cleaned.csv')
df_scaled = pd.read_csv('df_scaled.csv')

# Apply PCA on the scaled data
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

# Streamlit app title
st.title("Clustering For FIFA 19")

# Sidebar for clustering parameters
st.sidebar.header("Clustering Parameters")
clustering_method = st.sidebar.selectbox("Choose Clustering Method", ["K-means", "GMM", "Hierarchical", "DBSCAN", "Spectral"])

if clustering_method == "K-means":
    st.sidebar.header("K-means Parameters")
    n_clusters = st.sidebar.slider("Number of clusters (n_clusters)", min_value=2, max_value=10, value=3, step=1)
    init_method = st.sidebar.selectbox("Initialization method (init)", ["k-means++", "random"])
    n_init = st.sidebar.slider("Number of initializations (n_init)", min_value=10, max_value=50, value=29, step=1)
    max_iter = st.sidebar.slider("Maximum iterations (max_iter)", min_value=100, max_value=500, value=382, step=1)
    tol = st.sidebar.slider("Tolerance (tol)", min_value=1e-6, max_value=1e-2, value=1.9266922671867127e-05, format="%.8f", step=1e-6)
    algorithm = st.sidebar.selectbox("Algorithm", ["elkan", "lloyd"])

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
    df_clustered = df_cleaned.copy()
    df_clustered['Cluster_Kmeans'] = kmeans.labels_ + 1

    # Visualization of K-means clusters
    st.subheader("K-means Clustering on PCA Results")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='KMeans_Labels', data=df_pca, palette='viridis')
    plt.title('K-means Clustering on Principal Components')
    st.pyplot(plt)
    
    # Calculate and display silhouette score
    silhouette_avg = silhouette_score(df_pca[['PC1', 'PC2']], df_pca['KMeans_Labels'])
    st.subheader("Silhouette Score for K-means Clustering")
    st.write(f"Silhouette Score: {silhouette_avg:.4f}")
   

elif clustering_method == "GMM":
    st.sidebar.header("GMM Parameters")
    n_components = st.sidebar.slider("Number of components (n_components)", min_value=2, max_value=10, value=4, step=1)
    covariance_type = st.sidebar.selectbox("Covariance type", ["tied", "full", "diag", "spherical"])
    tol = st.sidebar.slider("Tolerance (tol)", min_value=1e-6, max_value=1e-2, value=0.0004118342756036605, format="%.8f", step=1e-6)
    max_iter = st.sidebar.slider("Maximum iterations (max_iter)", min_value=100, max_value=1000, value=776, step=1)

    # Run GMM clustering with selected parameters
    gmm = GaussianMixture(
        n_components=n_components, 
        covariance_type=covariance_type,
        tol=tol,
        max_iter=max_iter,
        random_state=42
    )
    gmm_labels = gmm.fit_predict(df_pca[['PC1', 'PC2']])
    df_pca['GMM_Labels'] = gmm_labels + 1
    df_clustered = df_cleaned.copy()
    df_clustered['Cluster_GMM'] = gmm_labels + 1

    # Visualization of GMM clusters
    st.subheader("GMM Clustering on PCA Results")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='GMM_Labels', data=df_pca, palette='viridis')
    plt.title('GMM Clustering on Principal Components')
    st.pyplot(plt)

    # Calculate and display silhouette score
    silhouette_avg = silhouette_score(df_pca[['PC1', 'PC2']], gmm_labels)
    st.subheader("Silhouette Score for GMM Clustering")
    st.write(f"Silhouette Score: {silhouette_avg:.4f}")

elif clustering_method == "Hierarchical":
    st.sidebar.header("Hierarchical Parameters")
    n_cluster_input = st.sidebar.slider("Number of clusters (n_clusters)", min_value=3, max_value=11, value=4, step=1)
    linkage_input = st.sidebar.selectbox("Linkage", ["average", "ward", "complete", "single"])
    metric_input = st.sidebar.selectbox("Metric", ["l1", "l2", "euclidean", "manhanttan", "cosine"])


    # Run Hierarchical clustering with selected parameters
    hierarchical = AgglomerativeClustering(
        n_clusters=n_cluster_input,
        linkage=linkage_input,
        metric=metric_input)
    hierarchical_labels = hierarchical.fit_predict(df_pca[['PC1', 'PC2']])
    df_pca['Hierarchical_Labels'] = hierarchical_labels + 1
    df_clustered = df_cleaned.copy()
    df_clustered['Cluster_Hierarchical'] = hierarchical_labels + 1

    # Visualization of Hierarchical clustering
    st.subheader("Hierarchical Clustering on PCA Results")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Hierarchical_Labels', data=df_pca, palette='viridis')
    plt.title('Hierarchical Clustering on Principal Components')
    st.pyplot(plt)

    # Calculate and display silhouette score
    silhouette_avg = silhouette_score(df_pca[['PC1', 'PC2']], hierarchical_labels)
    st.subheader("Silhouette Score for Hierarchical Clustering")
    st.write(f"Silhouette Score: {silhouette_avg:.4f}")

elif clustering_method == "DBSCAN":
    st.sidebar.header("DBSCAN Parameters")
    eps_input = st.sidebar.slider("Epsilon Values (eps)", min_value=0.1, max_value=5.0, value=2.065997806552059, format="%.1f", step=0.1)
    min_samples_input = st.sidebar.slider("Min Samples", min_value=4, max_value=20, value=16, step=1)
    algorithm_input = st.sidebar.selectbox("Algorithm", ["brute", "auto", "ball_tree", "kd_tree"])


    # Run Hierarchical clustering with selected parameters
    dbscan = DBSCAN(
        eps=eps_input, 
        min_samples=min_samples_input,
        algorithm=algorithm_input)

    dbscan_labels=dbscan.fit_predict(df_pca[['PC1', 'PC2']])
    df_pca['DBSCAN_Labels'] = dbscan_labels + 1
    df_clustered = df_cleaned.copy()
    df_clustered['Cluster_DBSCAN'] = dbscan_labels + 1

    # Visualization of DBSCAN clustering
    st.subheader("DBSCAN Clustering on PCA Results")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='DBSCAN_Labels', data=df_pca, palette='viridis', legend='full')
    plt.title('DBSCAN Clustering on Principal Components')
    st.pyplot(plt)

    # Calculate and display silhouette score
    silhouette_avg = silhouette_score(df_pca[['PC1', 'PC2']], dbscan_labels)
    st.subheader("Silhouette Score for DBSCAN Clustering")
    st.write(f"Silhouette Score: {silhouette_avg:.4f}")

elif clustering_method == "Spectral":
    st.sidebar.header("DBSCAN Parameters")
    affinity_input = st.sidebar.selectbox("Affinity", ["nearest_neighbors", "rbf"])

    # Run Hierarchical clustering with selected parameters
    spectral = SpectralClustering(n_clusters=4, affinity='nearest_neighbors', random_state=42)
    spectral_labels = spectral.fit_predict(df_pca[['PC1', 'PC2']])
    df_pca['Spectral_Labels'] = spectral_labels + 1
    df_clustered = df_cleaned.copy()
    df_clustered['Cluster_Spectral'] = spectral_labels + 1

    # Visualization of Spectral clustering
    st.subheader("Spectral Clustering on PCA Results")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Spectral_Labels', data=df_pca, palette='viridis')
    plt.title('Spectral Clustering on Principal Components')
    st.pyplot(plt)

    # Calculate and display silhouette score
    silhouette_avg = silhouette_score(df_pca[['PC1', 'PC2']], spectral_labels)
    st.subheader("Silhouette Score for Spectral Clustering")
    st.write(f"Silhouette Score: {silhouette_avg:.4f}")
  

# Display the number of records in each cluster
st.subheader("Number of Records in Each Cluster")
cluster_counts = df_clustered[df_clustered.columns[-1]].value_counts()
st.write(cluster_counts)

# Display mean statistics for each cluster
st.subheader("Mean Statistics for Each Cluster")
summary = df_clustered.groupby(df_clustered.columns[-1]).mean()
st.write(summary)

# Display median statistics for each cluster
st.subheader("Median Statistics for Each Cluster")
median_summary = df_clustered.groupby(df_clustered.columns[-1]).median()
st.write(median_summary)

# # Box plots for key features by clusters (hidden initially)
# #with st.expander("View Box Plots of Key Features by Cluster"):
# #    st.subheader("Box Plots of Key Features by Cluster")
#     key_features = df_clustered.columns.drop(df_clustered.columns[-1])
#     for feature in key_features:
#         plt.figure(figsize=(10, 6))
#         sns.boxplot(x=df_clustered.columns[-1], y=feature, data=df_clustered)
#         plt.title(f'Distribution of {feature} by Cluster')
#         st.pyplot(plt)
#         plt.close()  # Close the figure to free memory
