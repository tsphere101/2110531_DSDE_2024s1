import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(layout="wide")
st.title("Iris Dataset Analysis")


# Load and prepare data
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["Species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris.feature_names


df, feature_names = load_data()
X = df[feature_names].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar controls
st.sidebar.header("Analysis Controls")
clusters = st.sidebar.slider("Select Number of Clusters:", 1, 6, 3)

# 1. Feature Distribution Analysis
st.header("1. Feature Distributions by Species")

# Colors for species
colors = {"setosa": "#FF4B4B", "versicolor": "#4B4BFF", "virginica": "#4BFF4B"}

# Feature selection for box plot
feature = st.selectbox("Select feature for plotbox", df.columns)
fig = px.box(
    df,
    x="Species",
    y=feature,
    color="Species",
    color_discrete_map=colors,
    title=f"Distribution of {feature} by Species",
)
st.plotly_chart(fig)


# 2. Feature Relationships
st.header("2. Feature Relationships")
fig = px.scatter_matrix(
    df,
    title="Feature relationship by Species",
    dimensions=(col for col in df.columns if col != "Species"),
    color="Species",
    color_discrete_map=colors,
    height=800,
)
st.plotly_chart(fig)


# 3. Feature Correlations
st.header("3. Feature Correlations")
correlation = df[feature_names].corr()

# Create correlation heatmap
# Two decimal places text
# RdBu color scale
# Range from -1 to 1
fig = px.imshow(
    correlation,
    title="Feature Correlation Matrix",
    labels=dict(color="Correlation"),
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu",
    text_auto=".2f",
)
st.plotly_chart(fig)

# 4. Elbow Analysis
st.header("4. Elbow Analysis")


@st.cache_data
def perform_elbow_analysis(X, max_clusters=10):
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    return inertias


inertias = perform_elbow_analysis(X_scaled)

# Prepare the data for Plotly Express (convert inertias to a DataFrame)
elbow_df = pd.DataFrame(
    {"Number of Clusters": list(range(1, len(inertias) + 1)), "WCSS": inertias}
)

# Plotting the elbow plot using Plotly Express
fig = px.line(
    elbow_df,
    x="Number of Clusters",
    y="WCSS",
    title="Elbow Method Analysis",
    labels={
        "Number of Clusters": "Number of Clusters",
        "WCSS": "Inertia",
    },
    markers="o",
)

# Display the plot in Streamlit
st.plotly_chart(fig)

# 5. Clustering Analysis
st.header("5. Clustering Analysis")

# Perform clustering
kmeans = KMeans(n_clusters=clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
df["Cluster"] = cluster_labels.astype(str)

# Create comparison plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("Clustering Result")
    fig = px.scatter(
        df,
        x="petal width (cm)",
        y="petal length (cm)",
        color="Cluster",
        title="KMeans Clustering Result",
    )
    st.plotly_chart(fig)


with col2:
    st.subheader("Actual Species")
    fig = px.scatter(
        df,
        x="petal width (cm)",
        y="petal length (cm)",
        color="Species",
        title="KMeans Clustering Result",
        color_discrete_map=colors,
    )
    st.plotly_chart(fig)


# 6. Clustering Performance Analysis
st.header("6. Clustering Performance")
confusion_df = pd.crosstab(df["Species"], df["Cluster"], margins=True)
st.write("Confusion Matrix (Species vs Clusters):")
st.write(confusion_df)

# 7. Additional Statistics
st.header("7. Feature Statistics")
col3, col4 = st.columns(2)

with col3:
    st.subheader("Statistics by Species")
    species_stats = df.groupby("Species")[feature_names].agg(["mean", "std"]).round(2)
    st.write(species_stats)

with col4:
    st.subheader("Statistics by Cluster")
    cluster_stats = df.groupby("Cluster")[feature_names].agg(["mean", "std"]).round(2)
    st.write(cluster_stats)
