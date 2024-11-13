import os
import numpy as np
import matplotlib.pyplot as plt


def draw_uncertain_attentions(attentions: dict, path: str):
    
    os.makedirs(path, exist_ok=True)

    figure, axarr = plt.subplots(
        len(attentions.items()),
        2,
        figsize=(15 * 2, 15 * len(attentions.items())),
    )

    for i, (key, atts) in enumerate(attentions.items()):
        
        att, att_var = atts
        att_var = att_var / (att.abs() + 1e-7)
        att = att.detach().cpu().numpy().squeeze()
        att_var = att_var.detach().cpu().numpy().squeeze()
        
        ax = axarr[i, 0]
        cax = ax.imshow(att[0], cmap="Purples")
        ax.set_title(f"Attention for layer {key}")
        cbar = figure.colorbar(cax, orientation='horizontal')
        
        ax = axarr[i, 1]
        cax = ax.imshow(att_var[0], cmap="Purples")
        ax.set_title(f"Attention uncertainty for layer {key}")
        cbar = figure.colorbar(cax, orientation='horizontal')

    figure.savefig(f"{path}/attentions.png")

    print()
