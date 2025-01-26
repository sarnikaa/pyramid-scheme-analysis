import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations

# Title of the app
st.title("Pyramid Scheme Analysis.")

# File uploader widget
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the CSV data
    data = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.write("Data Preview:")
    st.dataframe(data.head())  # Display first few rows

    # Create the directed graph
    G = nx.DiGraph()

    # For each row, add edges based on depth and assign weights
    for i, row in data.iterrows():
        if row['depth_of_tree'] > 1:  # Assuming depth > 1 indicates a connection
            G.add_edge(row['depth_of_tree'] - 1, row['depth_of_tree'], weight=row['profit_markup'])

    # Section: Select Operation to Perform
    operation = st.sidebar.selectbox(
        "Select Operation",
        ("Graph Visualization", "Graph Properties", "Motif Analysis")
    )

    # Graph Visualization Section
    if operation == "Graph Visualization":
        st.header("Network Graph Visualization")

        # Select Layout for Graph
        layout = st.selectbox("Select Layout", ("Circular", "Spring", "Spectral"))
        plt.figure(figsize=(10, 7))

        if layout == "Circular":
            pos = nx.circular_layout(G)
        elif layout == "Spring":
            pos = nx.spring_layout(G)
        elif layout == "Spectral":
            pos = nx.spectral_layout(G)

        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', alpha=0.7)
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

        # Add edge weights as labels
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # Display the graph using Streamlit
        st.pyplot(plt)

    # Graph Properties Section
    elif operation == "Graph Properties":
        st.header("Graph Properties")

        # Select which property to display
        properties = st.multiselect(
            "Select Properties to Display",
            ["Density", "Average Distance", "Strongly Connected Components", "Transitivity", "Degree Centralization","All"]
        )

        if "Density" in properties:
            density = nx.density(G)
            st.metric(label="Density", value=f"{density:.4f}")

        if "Average Distance" in properties:
            if nx.is_strongly_connected(G):
                average_distance = nx.average_shortest_path_length(G)
                avg_distance_display = f"{average_distance:.4f}"
            else:
                avg_distance_display = "Undefined (Not Strongly Connected)"
            st.metric(label="Average Distance", value=avg_distance_display)

        if "Strongly Connected Components" in properties:
            strong_connectivity = list(nx.strongly_connected_components(G))  # List of components
            connectivity = len(strong_connectivity)  # Number of strongly connected components
            st.metric(label="Number of Strongly Connected Components", value=connectivity)

        if "Transitivity" in properties:
            transitivity = nx.transitivity(G.to_undirected())
            st.metric(label="Transitivity", value=f"{transitivity:.4f}")

        if "Degree Centralization" in properties:
            degree_centrality = nx.degree_centrality(G)
            max_degree = max(degree_centrality.values(), default=0)
            if len(G) > 2:
                degree_centralization = sum(max_degree - v for v in degree_centrality.values()) / (len(G) - 1) / (len(G) - 2)
                degree_centralization_display = f"{degree_centralization:.4f}"
            else:
                degree_centralization_display = "Undefined (Graph too small)"
            st.metric(label="Degree Centralization", value=degree_centralization_display)

        if "All" in properties:
            density = nx.density(G)
            st.metric(label="Density", value=f"{density:.4f}")

            if nx.is_strongly_connected(G):
                average_distance = nx.average_shortest_path_length(G)
                avg_distance_display = f"{average_distance:.4f}"
            else:
                avg_distance_display = "Undefined (Not Strongly Connected)"
            st.metric(label="Average Distance", value=avg_distance_display)

            strong_connectivity = list(nx.strongly_connected_components(G))  # List of components
            connectivity = len(strong_connectivity)  # Number of strongly connected components
            st.metric(label="Number of Strongly Connected Components", value=connectivity)

            transitivity = nx.transitivity(G.to_undirected())
            st.metric(label="Transitivity", value=f"{transitivity:.4f}")

            degree_centrality = nx.degree_centrality(G)
            max_degree = max(degree_centrality.values(), default=0)
            if len(G) > 2:
                degree_centralization = sum(max_degree - v for v in degree_centrality.values()) / (len(G) - 1) / (len(G) - 2)
                degree_centralization_display = f"{degree_centralization:.4f}"
            else:
                degree_centralization_display = "Undefined (Graph too small)"
            st.metric(label="Degree Centralization", value=degree_centralization_display)



    # Motif Analysis Section
    elif operation == "Motif Analysis":
        st.header("Motif Analysis")

        # Select Motif Size
        motif_size = st.selectbox("Select Motif Size", (2, 3, 4))

        # Function to count directed motifs
        def count_motifs(graph, size=2):
            motifs_count = {}
            # Generate all combinations of nodes of the specified size
            for nodes in combinations(graph.nodes(), size):
                subgraph = graph.subgraph(nodes)
                # Check if the subgraph has the appropriate number of edges
                if len(subgraph.edges()) == size - 1:  # Adjusted for size-2 motifs
                    motif = tuple(sorted(subgraph.edges()))  # Create a unique representation of the motif
                    if motif in motifs_count:
                        motifs_count[motif] += 1
                    else:
                        motifs_count[motif] = 1
            return motifs_count

        # Function to draw each motif
        def draw_motifs(motifs_count, title):
            plt.figure(figsize=(12, 6))
            num_motifs = len(motifs_count)
            if num_motifs == 0:
                st.write(f"No motifs found for {title}.")
                return

            for i, (motif, count) in enumerate(motifs_count.items(), 1):
                plt.subplot(1, num_motifs, i)
                motif_graph = nx.DiGraph()  # Directed graph for motif
                motif_graph.add_edges_from(motif)
                nx.draw(motif_graph, with_labels=True, node_color="skyblue", edge_color="grey", node_size=500)
                plt.title(f"Count: {count}")
            st.pyplot(plt)

        # Analyze motifs of the selected size
        directed_motifs = count_motifs(G, size=motif_size)

        # Display motifs and counts
        st.subheader(f"Motifs of Size {motif_size}")
        if not directed_motifs:
            st.write(f"No size-{motif_size} motifs found.")
        else:
            st.write(f"Motif counts for size {motif_size}:")
            st.write(pd.DataFrame([
                {"Motif": motif, "Count": count} for motif, count in directed_motifs.items()
            ]))
            draw_motifs(directed_motifs, f"Motifs of Size {motif_size}")

else:
    st.write("Please upload a CSV file to proceed.")
