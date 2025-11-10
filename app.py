import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
from datetime import datetime
from itertools import combinations
from collections import deque
import zipfile

# Configuration de la page
st.set_page_config(
    page_title="Karate Club Netw Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS moderne avec dark/light mode et signature
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .dark-mode {
        background-color: #0e1117;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .dark-mode .metric-card {
        background-color: #262730;
        color: white;
    }
    .export-section {
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .group-color {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        border: 1px solid #ccc;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #ddd;
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

class KarateApp:
    def __init__(self):
        if 'graph' not in st.session_state:
            st.session_state.graph = self.create_karate_graph()
            st.session_state.history = [st.session_state.graph.copy()]
            st.session_state.history_index = 0
        
        # Groupes flexibles - stockés dans session_state pour persistance
        if 'groups' not in st.session_state:
            st.session_state.groups = {
                "Turquoise (Instructeur)": "#40E0D0",
                "Orange (Propriétaire)": "orange"
            }
        
    def create_karate_graph(self):
        """Crée le graphe du club de karaté avec les nœuds"""
        G = nx.Graph()
        edges = [
            (1,12),(1,18),(1,6),(1,7),(1,11),(1,5),(1,13),(1,8),(1,4),(1,3),(1,14),
            (1,2),(1,32),(1,20),(1,22),(1,9),
            (2,4),(2,8),(2,14),(2,18),(2,20),(2,22),(2,3),(2,31),
            (3,4),(3,8),(3,9),(3,10),(3,14),(3,28),(3,29),(3,33),
            (4,8),(4,13),(4,14),
            (5,7),(5,11),
            (6,7),(6,11),(6,17),
            (7,17),
            (9,31),(9,33),(9,34),
            (10,34),
            (14,34),
            (15,33),(15,34),
            (16,34),(16,33),
            (19,33),(19,34),
            (20,34),
            (21,33),(21,34),
            (23,33),(23,34),
            (24,26),(24,28),(24,30),(24,33),(24,34),
            (25,26),(25,28),(25,32),
            (26,32),
            (27,34),(27,30),
            (28,34),
            (29,32),(29,34),
            (30,33),(30,34),
            (31,33),(31,34),
            (32,33),(32,34),
            (33,34)
        ]
        G.add_edges_from(edges)
        
        # Assigner les groupes
        turquoise_nodes = [1,2,3,4,5,6,7,8,9,11,12,13,14,17,18,20,22]
        for node in G.nodes():
            if node in turquoise_nodes:
                G.nodes[node]['group'] = "Turquoise (Instructeur)"
            else:
                G.nodes[node]['group'] = "Orange (Propriétaire)"
        
        return G
    
    def save_state(self):
        """Sauvegarde l'état actuel dans l'historique"""
        if len(st.session_state.history) > 20:  # Limiter la taille de l'historique
            st.session_state.history.pop(0)
        st.session_state.history.append(st.session_state.graph.copy())
        st.session_state.history_index = len(st.session_state.history) - 1
    
    def undo(self):
        """Retourne en arrière"""
        if st.session_state.history_index > 0:
            st.session_state.history_index -= 1
            st.session_state.graph = st.session_state.history[st.session_state.history_index].copy()
            st.rerun()
    
    def run(self):
        # Navigation entre pages
        page = st.sidebar.selectbox("📄 Navigation", 
                                   ["🏠 Visualisation Principale", "📊 Dashboard Analytique Complet"])
        
        if page == "🏠 Visualisation Principale":
            self.main_page()
        else:
            self.analytics_page()
        
        # Signature dans le footer
        st.markdown("""
        <div class="footer">
            Développé par Yasmine Aidoud - M2 ASD GRP 1
        </div>
        """, unsafe_allow_html=True)
    
    def main_page(self):
        """Page principale de visualisation"""
        # Header avec mode dark/light
        col_title, col_mode = st.columns([3, 1])
        with col_title:
            st.markdown('<h1 class="main-header">🥋 Karate Club Network Analyzer</h1>', 
                       unsafe_allow_html=True)
        with col_mode:
            dark_mode = st.toggle("🌙 Mode Sombre", value=False)
        
        # Appliquer le mode dark/light
        if dark_mode:
            st.markdown('<div class="dark-mode">', unsafe_allow_html=True)
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
        
        # Sidebar
        with st.sidebar:
            st.header("🎮 Contrôles du Réseau")
            
            # Navigation historique - SEULEMENT RETOUR
            st.subheader("⏪ Historique")
            if st.button("⏪ Retour", disabled=st.session_state.history_index == 0):
                self.undo()
            st.caption(f"État {st.session_state.history_index + 1}/{len(st.session_state.history)}")
            
            # Gestion des groupes - FLEXIBLE
            st.subheader("👥 Gestion des Groupes")
            
            # Affichage des groupes existants avec leurs couleurs
            if st.session_state.groups:
                st.write("**Groupes existants :**")
                for group_name, color in st.session_state.groups.items():
                    # Affichage avec cercle de couleur
                    st.markdown(f"<div style='display: flex; align-items: center; margin: 5px 0;'>"
                               f"<div class='group-color' style='background-color: {color};'></div>"
                               f"<span>{group_name} - <code>{color}</code></span>"
                               f"</div>", unsafe_allow_html=True)
            
            # Formulaire pour ajouter un nouveau groupe
            with st.form("add_group_form"):
                new_group = st.text_input("Nom du nouveau groupe", placeholder="Ex: Nouveau Groupe Bleu")
                new_color = st.color_picker("Couleur du groupe", "#ff6b6b")
                add_group_submitted = st.form_submit_button("➕ Ajouter Groupe")
                
                if add_group_submitted and new_group:
                    if new_group in st.session_state.groups:
                        st.error(f"❌ Le groupe '{new_group}' existe déjà!")
                    else:
                        st.session_state.groups[new_group] = new_color
                        st.success(f"✅ Groupe '{new_group}' ajouté avec la couleur {new_color}!")
                        st.rerun()
            
            # Option pour supprimer un groupe
            if len(st.session_state.groups) > 2:  # Au moins 3 groupes pour permettre la suppression
                group_to_delete = st.selectbox(
                    "Groupe à supprimer", 
                    [group for group in st.session_state.groups.keys() 
                     if group not in ["Turquoise (Instructeur)", "Orange (Propriétaire)"]]
                )
                if st.button("🗑️ Supprimer le Groupe", type="secondary"):
                    # Réaffecter les nœuds du groupe supprimé vers un groupe par défaut
                    deleted_group_nodes = [
                        node for node in st.session_state.graph.nodes() 
                        if st.session_state.graph.nodes[node].get('group') == group_to_delete
                    ]
                    
                    for node in deleted_group_nodes:
                        st.session_state.graph.nodes[node]['group'] = "Turquoise (Instructeur)"
                    
                    del st.session_state.groups[group_to_delete]
                    st.success(f"✅ Groupe '{group_to_delete}' supprimé!")
                    st.rerun()
            
            # Ajout de nœud
            st.subheader("🔘 Ajouter un Nœud")
            existing_nodes = sorted(list(st.session_state.graph.nodes()))  # TRI CROISSANT
            new_node_id = st.number_input("ID du nœud", min_value=1, step=1, value=35)
            
            # Liste déroulante des groupes avec TOUS les groupes disponibles et leurs couleurs
            selected_group = st.selectbox(
                "Groupe", 
                list(st.session_state.groups.keys()),  # Affiche tous les groupes
                format_func=lambda x: f"● {x}"  # Ajoute un cercle avant le nom
            )
            
            # Sélection des connexions - LISTE TRIÉE
            st.write("**Connexions :**")
            connections = st.multiselect(
                "Sélectionnez les nœuds connectés",
                existing_nodes,  # Déjà trié
                help="Choisissez tous les nœuds qui auront une relation avec le nouveau"
            )
            
            if st.button("✅ Ajouter le Nœud"):
                self.add_node(int(new_node_id), selected_group, connections)
            
            # Gestion des arêtes
            st.subheader("🔗 Gestion des Arêtes")
            col_edge1, col_edge2 = st.columns(2)
            with col_edge1:
                edge_node1 = st.selectbox("Nœud 1", existing_nodes)  # TRIÉ
            with col_edge2:
                # Filtrer pour éviter les doublons et trier
                available_nodes = sorted([n for n in existing_nodes if n != edge_node1])
                edge_node2 = st.selectbox("Nœud 2", available_nodes)  # TRIÉ
            
            col_add, col_remove = st.columns(2)
            with col_add:
                if st.button("➕ Ajouter Arête"):
                    self.add_edge(edge_node1, edge_node2)
            with col_remove:
                if st.button("➖ Supprimer Arête"):
                    self.remove_edge(edge_node1, edge_node2)
            
            # Suppression de nœud
            st.subheader("🗑️ Suppression")
            if existing_nodes:
                node_to_delete = st.selectbox("Nœud à supprimer", existing_nodes)  # TRIÉ
                if st.button("🔴 Supprimer Nœud", type="primary"):
                    self.delete_node(node_to_delete)
            else:
                st.write("Aucun nœud à supprimer")
        
        # Layout principal
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.display_network(dark_mode)
            
        with col2:
            self.display_quick_analytics()
        
        # Export des données
        st.markdown("---")
        self.display_export_section()
        
        if dark_mode:
            st.markdown('</div>', unsafe_allow_html=True)
    
    def get_all_groups_nodes(self):
        """Retourne un dictionnaire avec tous les groupes et leurs nœuds de façon flexible"""
        groups_nodes = {}
        for group_name in st.session_state.groups.keys():
            groups_nodes[group_name] = [
                node for node in st.session_state.graph.nodes() 
                if st.session_state.graph.nodes[node].get('group') == group_name
            ]
        return groups_nodes
    
    def analytics_page(self):
        """Page dashboard analytique complet"""
        st.markdown('<h1 class="main-header">📊 Dashboard Analytique Complet</h1>', 
                   unsafe_allow_html=True)
        
        # Calcul de toutes les métriques
        all_metrics = self.calculate_all_metrics()
        
        # Métriques globales
        st.header("📈 Métriques Globales du Réseau")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nœuds", all_metrics["ordre"])
        with col2:
            st.metric("Arêtes", all_metrics["taille"])
        with col3:
            st.metric("Densité", f"{nx.density(st.session_state.graph):.3f}")
        with col4:
            st.metric("Clustering Moyen", f"{all_metrics['coeff_moyen']:.3f}")
        
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            diameter = nx.diameter(st.session_state.graph) if nx.is_connected(st.session_state.graph) else "Non connecté"
            st.metric("Diamètre", diameter)
        with col6:
            st.metric("Composantes Connexes", nx.number_connected_components(st.session_state.graph))
        with col7:
            avg_degree = np.mean([d for _, d in st.session_state.graph.degree()])
            st.metric("Degré Moyen", f"{avg_degree:.2f}")
        with col8:
            triangles = sum(nx.triangles(st.session_state.graph).values()) // 3
            st.metric("Triangles", triangles)
        
        # Affichage des différentes analyses avec export
        tabs = st.tabs([
            "📊 Centralités", 
            "🎯 Clustering", 
            "🔍 Motifs & Cliques",
            "📈 K-core & Distribution",
            "👑 Analyse par Groupes"
        ])
        
        with tabs[0]:
            self.display_centralities_analysis(all_metrics)
        
        with tabs[1]:
            self.display_clustering_analysis(all_metrics)
        
        with tabs[2]:
            self.display_motifs_cliques_analysis(all_metrics)
        
        with tabs[3]:
            self.display_kcore_distribution_analysis(all_metrics)
        
        with tabs[4]:
            self.display_groups_analysis(all_metrics)
    
    def display_groups_analysis(self, metrics):
        """Affiche l'analyse flexible par groupes"""
        st.subheader("👥 Analyse par Groupes")
        
        # Récupération flexible de tous les groupes
        groups_nodes = self.get_all_groups_nodes()
        
        # Statistiques par groupe
        st.write("### 📊 Statistiques par Groupe")
        
        groups_stats = []
        for group_name, nodes in groups_nodes.items():
            if nodes:  # Vérifier que le groupe n'est pas vide
                group_degrees = [metrics["degree_dict"][node] for node in nodes]
                group_clustering = [metrics["clustering_dict"][node] for node in nodes]
                
                groups_stats.append({
                    "Groupe": group_name,
                    "Nœuds": len(nodes),
                    "Degré Moyen": f"{np.mean(group_degrees):.2f}",
                    "Clustering Moyen": f"{np.mean(group_clustering):.3f}",
                    "Densité Interne": f"{self.calculate_internal_density(nodes):.3f}"
                })
        
        if groups_stats:
            stats_df = pd.DataFrame(groups_stats)
            st.dataframe(stats_df, use_container_width=True)
        
        # Leaders par groupe
        st.write("### 👑 Leaders par Groupe")
        
        leaders_data = []
        for group_name, nodes in groups_nodes.items():
            if nodes:  # Vérifier que le groupe n'est pas vide
                # Trouver le leader par centralité de degré
                group_centralities = [(n, metrics["degree_centrality"][n]) for n in nodes]
                if group_centralities:
                    leader = max(group_centralities, key=lambda x: x[1])
                    leaders_data.append([group_name, leader[0], f"{leader[1]:.3f}"])
        
        if leaders_data:
            leaders_df = pd.DataFrame(leaders_data, columns=["Groupe", "Leader", "Centralité Degré"])
            st.dataframe(leaders_df, use_container_width=True)
        
        # Ponts entre groupes
        st.write("### 🌉 Ponts entre Groupes")
        
        bridges_data = []
        for group_name, nodes in groups_nodes.items():
            if nodes:  # Vérifier que le groupe n'est pas vide
                for node in nodes:
                    # Compter les connexions vers d'autres groupes
                    inter_connections = {}
                    for neighbor in st.session_state.graph.neighbors(node):
                        neighbor_group = st.session_state.graph.nodes[neighbor].get('group')
                        if neighbor_group != group_name:
                            inter_connections[neighbor_group] = inter_connections.get(neighbor_group, 0) + 1
                    
                    # Ajouter les ponts significatifs
                    for target_group, count in inter_connections.items():
                        if count > 0:
                            bridges_data.append([node, group_name, target_group, count])
        
        if bridges_data:
            bridges_df = pd.DataFrame(bridges_data, 
                                    columns=["Nœud", "Groupe Source", "Groupe Cible", "Connexions"])
            bridges_df = bridges_df.sort_values("Connexions", ascending=False)
            st.dataframe(bridges_df, use_container_width=True, height=300)
        else:
            st.info("Aucun pont identifié entre les groupes")
        
        # Visualisation des connexions inter-groupes
        st.write("### 🔗 Matrice des Connexions Inter-Groupes")
        
        connection_matrix = self.calculate_intergroup_connections()
        if connection_matrix is not None:
            st.dataframe(connection_matrix, use_container_width=True)
        
        # Export section
        with st.expander("📤 Exporter l'Analyse par Groupes"):
            if groups_stats:
                self.export_dataframe(pd.DataFrame(groups_stats), "statistiques_groupes")
            if leaders_data:
                self.export_dataframe(leaders_df, "leaders_groupes")
            if bridges_data:
                self.export_dataframe(bridges_df, "ponts_intergroupes")
    
    def calculate_internal_density(self, nodes):
        """Calcule la densité interne d'un groupe de nœuds"""
        if len(nodes) < 2:
            return 0.0
        
        subgraph = st.session_state.graph.subgraph(nodes)
        possible_edges = len(nodes) * (len(nodes) - 1) / 2
        actual_edges = subgraph.number_of_edges()
        
        return actual_edges / possible_edges if possible_edges > 0 else 0.0
    
    def calculate_intergroup_connections(self):
        """Calcule la matrice des connexions entre groupes"""
        groups_nodes = self.get_all_groups_nodes()
        group_names = list(groups_nodes.keys())
        
        # Créer une matrice vide
        connection_matrix = pd.DataFrame(0, index=group_names, columns=group_names)
        
        # Compter les arêtes entre groupes
        for edge in st.session_state.graph.edges():
            node1, node2 = edge
            group1 = st.session_state.graph.nodes[node1].get('group')
            group2 = st.session_state.graph.nodes[node2].get('group')
            
            if group1 != group2 and group1 in group_names and group2 in group_names:
                connection_matrix.loc[group1, group2] += 1
                connection_matrix.loc[group2, group1] += 1  # Graphe non orienté
        
        return connection_matrix if not connection_matrix.empty else None

    def calculate_all_metrics(self):
        """Calcule toutes les métriques du notebook"""
        G = st.session_state.graph
        nodes = list(G.nodes())
        edges = list(G.edges())
        
        # Ordre et taille
        ordre = G.number_of_nodes()
        taille = G.number_of_edges()
        
        # Degrés
        degree_dict = {node: 0 for node in nodes}
        for u, v in edges:
            degree_dict[u] += 1
            degree_dict[v] += 1
        
        # Clustering
        clustering_dict = {}
        for node in nodes:
            voisins = list(G.neighbors(node))
            k_v = len(voisins)
            if k_v < 2:
                clustering_dict[node] = 0.0
                continue
            e_v = 0
            for i in range(len(voisins)):
                for j in range(i + 1, len(voisins)):
                    if G.has_edge(voisins[i], voisins[j]):
                        e_v += 1
            cc_v = (2 * e_v) / (k_v * (k_v - 1))
            clustering_dict[node] = round(cc_v, 3)
        
        coeff_moyen = round(np.mean(list(clustering_dict.values())), 3)
        
        # Cliques
        cliques = list(nx.find_cliques(G))
        max_clique = sorted(max(cliques, key=len))
        taille_max_clique = len(max_clique)
        
        # K-core
        core_numbers = nx.core_number(G)
        k_core_max = max(core_numbers.values())
        
        # Centralités
        degree_centrality = {v: degree_dict[v] / (ordre - 1) for v in nodes}
        
        def closeness_centrality_manual(G, node):
            distances = nx.single_source_shortest_path_length(G, node)
            total = sum(distances.values())
            if total > 0 and len(distances) > 1:
                return (len(distances) - 1) / total
            else:
                return 0.0
        
        closeness_dict = {v: round(closeness_centrality_manual(G, v), 4) for v in nodes}
        
        # Centralité d'intermédiarité (version simplifiée)
        betweenness_dict = nx.betweenness_centrality(G)
        
        # Centralité HITS
        try:
            authority_dict, hub_dict = nx.hits(G)
        except:
            # Fallback si HITS échoue
            authority_dict = {v: 0.0 for v in nodes}
            hub_dict = {v: 0.0 for v in nodes}
        
        return {
            "ordre": ordre,
            "taille": taille,
            "degree_dict": degree_dict,
            "clustering_dict": clustering_dict,
            "coeff_moyen": coeff_moyen,
            "cliques": cliques,
            "max_clique": max_clique,
            "taille_max_clique": taille_max_clique,
            "core_numbers": core_numbers,
            "k_core_max": k_core_max,
            "degree_centrality": degree_centrality,
            "closeness_dict": closeness_dict,
            "betweenness_dict": betweenness_dict,
            "authority_dict": authority_dict,
            "hub_dict": hub_dict,
            "nodes": nodes
        }
    
    def display_centralities_analysis(self, metrics):
        """Affiche l'analyse des centralités avec export"""
        st.subheader("🎯 Analyse des Centralités")
        
        # Création du DataFrame des centralités
        centrality_df = pd.DataFrame({
            "Sommet": metrics["nodes"],
            "Groupe": [st.session_state.graph.nodes[node].get('group', 'Non assigné') for node in metrics["nodes"]],
            "Degré": [metrics["degree_centrality"][v] for v in metrics["nodes"]],
            "Proximité": [metrics["closeness_dict"][v] for v in metrics["nodes"]],
            "Intermédiarité": [metrics["betweenness_dict"][v] for v in metrics["nodes"]],
            "Authority": [metrics["authority_dict"][v] for v in metrics["nodes"]],
            "Hub": [metrics["hub_dict"][v] for v in metrics["nodes"]],
        }).sort_values(by="Degré", ascending=False)
        
        # Affichage du tableau
        st.dataframe(centrality_df, use_container_width=True, height=400)
        
        # Top 3 par type de centralité
        st.subheader("🏆 Top 3 par Type de Centralité")
        
        cols = st.columns(5)
        centralities = ["Degré", "Proximité", "Intermédiarité", "Authority", "Hub"]
        
        for i, col in enumerate(cols):
            with col:
                st.write(f"**{centralities[i]}**")
                top_3 = centrality_df.nlargest(3, centralities[i])[["Sommet", "Groupe", centralities[i]]]
                for _, row in top_3.iterrows():
                    st.write(f"• {int(row['Sommet'])} ({row['Groupe']}): {row[centralities[i]]:.3f}")
        
        # Graphique comparatif
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metrics["nodes"]))
        width = 0.15
        
        # Normalisation pour le graphique
        for col in centralities:
            centrality_df[col] = centrality_df[col] / centrality_df[col].max()
        
        plt.bar(x - 2*width, centrality_df["Degré"], width=width, label="Degré")
        plt.bar(x - width, centrality_df["Proximité"], width=width, label="Proximité")
        plt.bar(x, centrality_df["Intermédiarité"], width=width, label="Intermédiarité")
        plt.bar(x + width, centrality_df["Authority"], width=width, label="Authority")
        plt.bar(x + 2*width, centrality_df["Hub"], width=width, label="Hub")
        
        plt.xticks(x, centrality_df["Sommet"])
        plt.xlabel("Sommets")
        plt.ylabel("Valeur Normalisée")
        plt.title("Comparaison des Centralités (Normalisées)")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Export section
        with st.expander("📤 Exporter les Analyses de Centralités"):
            col1, col2 = st.columns(2)
            with col1:
                self.export_dataframe(centrality_df, "centralites_completes")
            with col2:
                self.export_plot(fig, "comparaison_centralites")
    
    def display_clustering_analysis(self, metrics):
        """Affiche l'analyse du clustering avec export"""
        st.subheader("🔄 Analyse du Clustering")
        
        # DataFrame du clustering
        clustering_df = pd.DataFrame({
            "Sommet": metrics["nodes"],
            "Groupe": [st.session_state.graph.nodes[node].get('group', 'Non assigné') for node in metrics["nodes"]],
            "Clustering": [metrics["clustering_dict"][v] for v in metrics["nodes"]],
            "Degré": [metrics["degree_dict"][v] for v in metrics["nodes"]]
        }).sort_values(by="Clustering", ascending=False)
        
        st.dataframe(clustering_df, use_container_width=True, height=400)
        
        # Graphique clustering vs degré
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogramme du clustering
        ax1.hist(clustering_df["Clustering"], bins=15, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Coefficient de Clustering')
        ax1.set_ylabel('Nombre de nœuds')
        ax1.set_title('Distribution du Clustering')
        ax1.grid(True, alpha=0.3)
        
        # Clustering vs Degré
        ax2.scatter(clustering_df["Degré"], clustering_df["Clustering"], alpha=0.6)
        ax2.set_xlabel('Degré')
        ax2.set_ylabel('Coefficient de Clustering')
        ax2.set_title('Clustering vs Degré')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Export section
        with st.expander("📤 Exporter les Analyses de Clustering"):
            col1, col2 = st.columns(2)
            with col1:
                self.export_dataframe(clustering_df, "clustering_analysis")
            with col2:
                self.export_plot(fig, "clustering_vs_degre")
    
    def display_motifs_cliques_analysis(self, metrics):
        """Affiche l'analyse des motifs et cliques avec export"""
        st.subheader("🔍 Analyse des Motifs et Cliques")
        
        # Informations sur les cliques
        st.write(f"**Nombre total de cliques :** {len(metrics['cliques'])}")
        st.write(f"**Plus grande clique :** {metrics['max_clique']} (taille: {metrics['taille_max_clique']})")
        
        # Tableau des cliques
        cliques_df = pd.DataFrame({
            "Clique": [', '.join(map(str, sorted(c))) for c in metrics["cliques"]],
            "Taille": [len(c) for c in metrics["cliques"]]
        }).sort_values(by="Taille", ascending=False).reset_index(drop=True)
        
        st.dataframe(cliques_df, use_container_width=True, height=300)
        
        # Distribution des tailles de cliques
        fig, ax = plt.subplots(figsize=(10, 4))
        clique_sizes = [len(c) for c in metrics["cliques"]]
        ax.hist(clique_sizes, bins=range(2, max(clique_sizes) + 2), alpha=0.7, edgecolor='black')
        ax.set_xlabel('Taille des Cliques')
        ax.set_ylabel('Nombre')
        ax.set_title('Distribution des Tailles de Cliques')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Export section
        with st.expander("📤 Exporter les Analyses de Cliques"):
            col1, col2 = st.columns(2)
            with col1:
                self.export_dataframe(cliques_df, "cliques_analysis")
            with col2:
                self.export_plot(fig, "distribution_cliques")
    
    def display_kcore_distribution_analysis(self, metrics):
        """Affiche l'analyse K-core et distribution avec export"""
        st.subheader("📈 Analyse K-core et Distribution")
        
        # DataFrame K-core
        kcore_df = pd.DataFrame({
            "Sommet": metrics["nodes"],
            "Groupe": [st.session_state.graph.nodes[node].get('group', 'Non assigné') for node in metrics["nodes"]],
            "K-core": [metrics["core_numbers"][v] for v in metrics["nodes"]],
            "Degré": [metrics["degree_dict"][v] for v in metrics["nodes"]]
        }).sort_values(by="K-core", ascending=False)
        
        st.dataframe(kcore_df, use_container_width=True, height=400)
        
        # Graphiques
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Distribution K-core
        ax1.hist(kcore_df["K-core"], bins=15, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Valeur K-core')
        ax1.set_ylabel('Nombre de nœuds')
        ax1.set_title('Distribution des K-cores')
        ax1.grid(True, alpha=0.3)
        
        # Distribution des degrés
        degree_sequence = sorted([metrics["degree_dict"][n] for n in metrics["nodes"]], reverse=True)
        ax2.hist(degree_sequence, bins=15, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Degré')
        ax2.set_ylabel('Nombre de nœuds')
        ax2.set_title('Distribution des Degrés')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Export section
        with st.expander("📤 Exporter les Analyses K-core"):
            col1, col2 = st.columns(2)
            with col1:
                self.export_dataframe(kcore_df, "kcore_analysis")
            with col2:
                self.export_plot(fig, "kcore_distribution")

    
    def add_node(self, node_id, group, connections):
        """Ajoute un nœud avec ses connexions"""
        if node_id in st.session_state.graph.nodes():
            st.error(f"❌ Le nœud {node_id} existe déjà!")
            return
        
        self.save_state()
        st.session_state.graph.add_node(node_id, group=group)
        
        for target_node in connections:
            st.session_state.graph.add_edge(node_id, target_node)
        
        st.success(f"✅ Nœud {node_id} ajouté avec {len(connections)} connexions!")
        st.rerun()
    
    def add_edge(self, node1, node2):
        """Ajoute une arête entre deux nœuds"""
        if st.session_state.graph.has_edge(node1, node2):
            st.error("❌ Cette arête existe déjà!")
            return
        
        self.save_state()
        st.session_state.graph.add_edge(node1, node2)
        st.success(f"✅ Arête ({node1}-{node2}) ajoutée!")
        st.rerun()
    
    def remove_edge(self, node1, node2):
        """Supprime une arête spécifique"""
        if not st.session_state.graph.has_edge(node1, node2):
            st.error("❌ Cette arête n'existe pas!")
            return
        
        self.save_state()
        st.session_state.graph.remove_edge(node1, node2)
        st.success(f"✅ Arête ({node1}-{node2}) supprimée!")
        st.rerun()
    
    def delete_node(self, node_id):
        """Supprime un nœud et toutes ses connexions"""
        connections_count = len(list(st.session_state.graph.edges(node_id)))
        self.save_state()
        st.session_state.graph.remove_node(node_id)
        st.success(f"🗑️ Nœud {node_id} et ses {connections_count} connexions supprimés!")
        st.rerun()
    
    def display_network(self, dark_mode=False):
        """Affiche le réseau"""
        st.subheader("🌐 Visualisation du Réseau")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(st.session_state.graph, seed=42)
        
        # Couleurs par groupe
        node_colors = []
        for node in st.session_state.graph.nodes():
            group = st.session_state.graph.nodes[node].get('group', 'Turquoise (Instructeur)')
            node_colors.append(st.session_state.groups.get(group, "#cccccc"))
        
        # Dessin avec effet néon en mode dark
        if dark_mode:
            # Arêtes blanches pour mode sombre
            nx.draw_networkx_edges(st.session_state.graph, pos, alpha=0.7, width=1.5, 
                                 edge_color='white', ax=ax)
            # Nœuds avec effet glow
            nx.draw_networkx_nodes(st.session_state.graph, pos, node_color=node_colors, 
                                 node_size=600, ax=ax, edgecolors='white', 
                                 linewidths=2, alpha=0.9)
        else:
            nx.draw_networkx_edges(st.session_state.graph, pos, alpha=0.5, width=1.2, ax=ax)
            nx.draw_networkx_nodes(st.session_state.graph, pos, node_color=node_colors, 
                                 node_size=500, ax=ax, edgecolors='black', linewidths=1)
        
        nx.draw_networkx_labels(st.session_state.graph, pos, font_size=8, ax=ax)
        
        ax.set_title("Réseau du Club de Karaté de Zachary", fontsize=16)
        ax.axis('off')
        
        st.pyplot(fig)
        
        # Export de la visualisation
        with st.expander("📤 Exporter cette Visualisation"):
            self.export_plot(fig, "reseau_karate")
    
    def display_quick_analytics(self):
        """Affiche les analyses rapides sur la page principale"""
        st.subheader("📊 Aperçu Analytique")
        
        # Métriques
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nœuds", st.session_state.graph.number_of_nodes())
        with col2:
            st.metric("Arêtes", st.session_state.graph.number_of_edges())
        
        # Top centralités global
        st.write("**🎯 Top 3 - Centralité de Degré**")
        degree_centrality = nx.degree_centrality(st.session_state.graph)
        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for node, centrality in top_nodes:
            group = st.session_state.graph.nodes[node].get('group', 'Non assigné')
            st.write(f"• Nœud {node} ({group}): `{centrality:.3f}`")
        
        st.write("---")
        st.write("ℹ️ *Pour plus d'analyses, visitez le* **📊 Dashboard Analytique Complet**")
    
    def display_export_section(self):
        """Affiche la section d'export"""
        st.subheader("📤 Export Complet des Données")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Exporter Tous les CSV", use_container_width=True):
                self.export_all_csv()
        
        with col2:
            if st.button("🖼️ Exporter Toutes les Images", use_container_width=True):
                self.export_all_images()
        
        with col3:
            if st.button("📊 Exporter Rapport Complet", use_container_width=True):
                self.export_full_report()
    
    def export_dataframe(self, df, filename):
        """Exporte un DataFrame en CSV"""
        csv = df.to_csv(index=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label=f"📥 {filename}.csv",
            data=csv,
            file_name=f"{filename}_{timestamp}.csv",
            mime="text/csv",
            key=f"csv_{filename}_{timestamp}"
        )
    
    def export_plot(self, fig, filename):
        """Exporte un graphique en image"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label=f"📥 {filename}.png",
            data=buf,
            file_name=f"{filename}_{timestamp}.png",
            mime="image/png",
            key=f"img_{filename}_{timestamp}"
        )
    
    def export_all_csv(self):
        """Exporte tous les CSV dans un ZIP"""
        try:
            metrics = self.calculate_all_metrics()
            
            # Création de tous les DataFrames
            centrality_df = pd.DataFrame({
                "Sommet": metrics["nodes"],
                "Groupe": [st.session_state.graph.nodes[node].get('group', 'Non assigné') for node in metrics["nodes"]],
                "Degré": [metrics["degree_centrality"][v] for v in metrics["nodes"]],
                "Proximité": [metrics["closeness_dict"][v] for v in metrics["nodes"]],
                "Intermédiarité": [metrics["betweenness_dict"][v] for v in metrics["nodes"]],
                "Authority": [metrics["authority_dict"][v] for v in metrics["nodes"]],
                "Hub": [metrics["hub_dict"][v] for v in metrics["nodes"]],
            })
            
            clustering_df = pd.DataFrame({
                "Sommet": metrics["nodes"],
                "Groupe": [st.session_state.graph.nodes[node].get('group', 'Non assigné') for node in metrics["nodes"]],
                "Clustering": [metrics["clustering_dict"][v] for v in metrics["nodes"]],
                "Degré": [metrics["degree_dict"][v] for v in metrics["nodes"]]
            })
            
            kcore_df = pd.DataFrame({
                "Sommet": metrics["nodes"],
                "Groupe": [st.session_state.graph.nodes[node].get('group', 'Non assigné') for node in metrics["nodes"]],
                "K-core": [metrics["core_numbers"][v] for v in metrics["nodes"]],
                "Degré": [metrics["degree_dict"][v] for v in metrics["nodes"]]
            })
            
            cliques_df = pd.DataFrame({
                "Clique": [', '.join(map(str, sorted(c))) for c in metrics["cliques"]],
                "Taille": [len(c) for c in metrics["cliques"]]
            })
            
            # Données des nœuds et arêtes
            nodes_data = []
            for node in st.session_state.graph.nodes():
                nodes_data.append({
                    'Nœud': node,
                    'Groupe': st.session_state.graph.nodes[node].get('group', 'Non assigné'),
                    'Degré': st.session_state.graph.degree(node)
                })
            nodes_df = pd.DataFrame(nodes_data)
            
            edges_data = []
            for edge in st.session_state.graph.edges():
                edges_data.append({'Source': edge[0], 'Cible': edge[1]})
            edges_df = pd.DataFrame(edges_data)
            
            # Matrice d'adjacence simplifiée (évite scipy)
            nodes_sorted = sorted(st.session_state.graph.nodes())
            adj_data = []
            for i, node1 in enumerate(nodes_sorted):
                row = {}
                for j, node2 in enumerate(nodes_sorted):
                    row[node2] = 1 if st.session_state.graph.has_edge(node1, node2) else 0
                adj_data.append(row)
            
            adj_df = pd.DataFrame(adj_data, index=nodes_sorted)
            adj_df.index.name = 'Nœud'
            
            # Analyse par groupes (nouvelle)
            groups_nodes = self.get_all_groups_nodes()
            groups_stats = []
            for group_name, nodes in groups_nodes.items():
                if nodes:
                    group_degrees = [metrics["degree_dict"][node] for node in nodes]
                    group_clustering = [metrics["clustering_dict"][node] for node in nodes]
                    
                    groups_stats.append({
                        "Groupe": group_name,
                        "Nœuds": len(nodes),
                        "Degré Moyen": f"{np.mean(group_degrees):.2f}",
                        "Clustering Moyen": f"{np.mean(group_clustering):.3f}",
                        "Densité Interne": f"{self.calculate_internal_density(nodes):.3f}"
                    })
            
            groups_df = pd.DataFrame(groups_stats) if groups_stats else pd.DataFrame()
            
            # Création du ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                zip_file.writestr('centralites.csv', centrality_df.to_csv(index=False))
                zip_file.writestr('clustering.csv', clustering_df.to_csv(index=False))
                zip_file.writestr('kcore.csv', kcore_df.to_csv(index=False))
                zip_file.writestr('cliques.csv', cliques_df.to_csv(index=False))
                zip_file.writestr('noeuds.csv', nodes_df.to_csv(index=False))
                zip_file.writestr('aretes.csv', edges_df.to_csv(index=False))
                zip_file.writestr('matrice_adjacence.csv', adj_df.to_csv())
                if not groups_df.empty:
                    zip_file.writestr('statistiques_groupes.csv', groups_df.to_csv(index=False))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📥 Télécharger Tous les CSV (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"karate_club_analyses_completes_{timestamp}.zip",
                mime="application/zip"
            )
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'export : {str(e)}")
    
    def export_all_images(self):
        """Exporte toutes les images dans un ZIP"""
        st.info("🖼️ Utilisez les boutons d'export individuels dans chaque section pour les graphiques spécifiques")
    
    def export_full_report(self):
        """Exporte un rapport complet avec toutes les analyses"""
        metrics = self.calculate_all_metrics()
        
        report = "RAPPORT COMPLET - CLUB DE KARATÉ DE ZACHARY\n"
        report += "=" * 50 + "\n\n"
        
        # Métriques globales
        report += "MÉTRIQUES GLOBALES:\n"
        report += f"- Nœuds: {metrics['ordre']}\n"
        report += f"- Arêtes: {metrics['taille']}\n"
        report += f"- Densité: {nx.density(st.session_state.graph):.3f}\n"
        report += f"- Coefficient de clustering moyen: {metrics['coeff_moyen']:.3f}\n"
        report += f"- Diamètre: {nx.diameter(st.session_state.graph) if nx.is_connected(st.session_state.graph) else 'Non connecté'}\n"
        report += f"- Composantes connexes: {nx.number_connected_components(st.session_state.graph)}\n"
        report += f"- Degré moyen: {np.mean([d for _, d in st.session_state.graph.degree()]):.2f}\n"
        report += f"- Plus grande clique: {metrics['max_clique']} (taille: {metrics['taille_max_clique']})\n"
        report += f"- K-core maximal: {metrics['k_core_max']}\n\n"
        
        # Analyse par groupes
        report += "ANALYSE PAR GROUPES:\n"
        groups_nodes = self.get_all_groups_nodes()
        for group_name, nodes in groups_nodes.items():
            if nodes:
                group_degrees = [metrics["degree_dict"][node] for node in nodes]
                group_clustering = [metrics["clustering_dict"][node] for node in nodes]
                
                report += f"\n{group_name}:\n"
                report += f"  - Nœuds: {len(nodes)}\n"
                report += f"  - Degré moyen: {np.mean(group_degrees):.2f}\n"
                report += f"  - Clustering moyen: {np.mean(group_clustering):.3f}\n"
                report += f"  - Densité interne: {self.calculate_internal_density(nodes):.3f}\n"
                
                # Leader du groupe
                group_centralities = [(n, metrics["degree_centrality"][n]) for n in nodes]
                if group_centralities:
                    leader = max(group_centralities, key=lambda x: x[1])
                    report += f"  - Leader: Nœud {leader[0]} (centralité: {leader[1]:.3f})\n"
        
        # Top centralités
        report += "\nTOP CENTRALITÉS:\n"
        centrality_types = [
            ("DEGRÉ", metrics["degree_centrality"]),
            ("PROXIMITÉ", metrics["closeness_dict"]),
            ("INTERMÉDIARITÉ", metrics["betweenness_dict"]),
            ("AUTHORITY", metrics["authority_dict"]),
            ("HUB", metrics["hub_dict"])
        ]
        
        for name, centrality_dict in centrality_types:
            top_3 = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:3]
            report += f"{name}:\n"
            for node, value in top_3:
                group = st.session_state.graph.nodes[node].get('group', 'Non assigné')
                report += f"  • Nœud {node} ({group}): {value:.3f}\n"
            report += "\n"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📥 Télécharger Rapport Complet (TXT)",
            data=report,
            file_name=f"rapport_karate_club_complet_{timestamp}.txt",
            mime="text/plain"
        )

# Lancement
if __name__ == "__main__":
    app = KarateApp()
    app.run()