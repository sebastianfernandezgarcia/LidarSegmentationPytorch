import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

todos_dataframes = []
"""
relaciones = {
    0: "Sin clasificar",
    1: "Suelo",
    2: "Edificio",
    3: "Señales",
    4: "Bolardo",
    5: "Papelera",
    6: "Barrier",
    7: "Peaton",
    8: "Coche",
    9: "Vegetación"
    }
"""

relaciones = {0: 'Suelo', 1: 'Edificio', 2: 'Señales', 3: 'Bolardo', 4: 'Papelera', 5: 'Barrier', 6: 'Peaton', 7: 'Coche', 8: 'Vegetación'}



def muestra_matriz_confusion(cf_matrix):
        # set the figure size
        fig, ax = plt.subplots(figsize=(12, 12))

        # plot the confusion matrix as an image
        im = ax.imshow(cf_matrix, cmap=plt.cm.Blues)

        # add a colorbar
        cbar = ax.figure.colorbar(im, ax=ax)

        # set the tick labels
        ax.set_xticks(np.arange(cf_matrix.shape[1]))
        ax.set_yticks(np.arange(cf_matrix.shape[0]))

        # set the tick labels' font size
        ax.tick_params(axis='both', which='major', labelsize=10)

        # set the tick labels' position

        ax.set_xticklabels(['{}'.format(relaciones[i]) for i in range(cf_matrix.shape[1])])
        ax.set_yticklabels(['{}'.format(relaciones[i]) for i in range(cf_matrix.shape[1])])

        # set the tick labels' orientation
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # add the counts to each cell
        for i in range(cf_matrix.shape[0]):
            for j in range(cf_matrix.shape[1]):
                ax.text(j, i, format(cf_matrix[i, j], ','),
                        ha="center", va="center", color="black", fontsize=10)

        # set the title
        ax.set_title("Confusion Matrix", fontsize=16)

        # set the axis labels
        ax.set_xlabel("Predicted Class", fontsize=14)
        ax.set_ylabel("True Class", fontsize=14)

        # show the plot
        plt.tight_layout()

        plt.show(block=False)
        plt.savefig('confusion_matrix.png')
        plt.pause(5)
        plt.close()
        #plt.show()