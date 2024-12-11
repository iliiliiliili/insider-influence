import os
import numpy as np
import matplotlib.pyplot as plt


def draw_uncertain_attentions(attentions: dict, path: str):

    os.makedirs(path, exist_ok=True)

    figure, axarr = plt.subplots(
        2,
        len(attentions.items()),
        figsize=(8 * len(attentions.items()), 8 * 2),
    )

    for i, (key, atts) in enumerate(attentions.items()):

        att, att_var = atts
        att_var = att_var / (att.abs() + 1e-7)
        att = att.detach().cpu().numpy().squeeze()
        att_var = att_var.detach().cpu().numpy().squeeze()

        ax = axarr[0, i]
        cax = ax.imshow(att[0], cmap="Purples")
        ax.set_title(f"Attention for layer {key}")
        cbar = figure.colorbar(cax, orientation="horizontal")

        ax = axarr[1, i]
        cax = ax.imshow(att_var[0], cmap="Purples")
        ax.set_title(f"Attention uncertainty for layer {key}")
        cbar = figure.colorbar(cax, orientation="horizontal")

    figure.savefig(f"{path}/attentions.png")

    print()


def draw_uncertain_attention_graphs(
    attentions: dict,
    adjacencies: np.array,
    path: str,
    max_width=10,
):

    import networkx as nx
    
    os.makedirs(path, exist_ok=True)

    graph = nx.MultiGraph()

    node_count = attentions[0][0].shape[-1]

    for n in range(node_count):
        graph.add_node(n)

    for b in range(attentions[0][0].shape[0]):
        pos = None

        figure, axarr = plt.subplots(
            len(attentions.items()),
            2,
            figsize=(4 * 2, 4 * len(attentions.items())),
        )

        for i, (key, atts) in enumerate(attentions.items()):

            att, att_var = atts
            att_var = att_var / (att.abs() + 1e-7)
            att = att.detach().cpu().numpy().squeeze()
            att_var = att_var.detach().cpu().numpy().squeeze()

            for p in range(2):

                ax = axarr[i, p]

                graph.clear_edges()

                min_att = att[b].min()
                max_att = att[b].max()

                for k in range(len(att[1])):
                    for q in range(k + 1, len(att[1])):

                        attention_width = (
                            max_width * (att[b, k, q] - min_att) / (max_att - min_att)
                        )
                        variance_width = (
                            max_width
                            * (att_var[b, k, q] - min_att)
                            / (max_att - min_att)
                        )

                        if p == 0:

                            graph.add_edge(
                                k,
                                q,
                                # weight=attention_width,
                                width=attention_width,
                                color="blue",
                            )
                            
                            if adjacencies[b][k][q] > 0:
                                graph.add_edge(
                                    k,
                                    q,
                                    width=attention_width / 2 if attention_width > 0.5 else 0.5,
                                    color="yellow",
                                )
                        else:
                            graph.add_edge(k, q, width=variance_width, color="red")

                if pos is None:
                    pos = nx.spring_layout(graph)
                    pos[node_count - 1] = np.array([0.0, 0.0], dtype=np.float32)

                colors = []
                widths = []

                for u, v, attrib_dict in list(graph.edges.data()):
                    colors.append(attrib_dict["color"])
                    widths.append(attrib_dict["width"])

                node_colors = [
                    "blue" if n + 1 < 50 else "purple" for n in graph.nodes()
                ]

                if p == 0:
                    ax.set_title(f"Attention for layer {key}")
                else:
                    ax.set_title(f"Attention uncertainty for layer {key}")

                nx.draw(
                    graph,
                    pos,
                    edge_color=colors,
                    width=widths,
                    node_color=node_colors,
                    node_size=100,
                    ax=ax,
                )

        figure.savefig(f"{path}/attentions_graph_{b}.png")
        print(f"Saved figure {path}/attentions_graph_{b}.png")

        print()
