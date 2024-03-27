import json
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

data = {
    "Motif basis": {
        "ERM": {
            "original": {
                "id_val": 0.934,
                "val": 0.628,
                "test": 0.651,
                "wiou": np.nan
            }
        },
        "LECI": {
            "original": {
                "id_val": 0.913,
                "val": 0.867,
                "test": 0.823,
                "wiou": 0.781
            },
            "permuted": {
                "id_val": 0.35,
                "val": 0.536,
                "test": 0.419,
            },
        },
        "CIGA": {
            "original": {
                "id_val": 0.92,
                "val": 0.817,
                "test": 0.504,
                "wiou": 0.673
            },
            "permuted": {
                "id_val": 0.739,
                "val": 0.417,
                "test": 0.454,
            },
        },
        "GSAT": {
            "original": {
                "id_val": 0.930,
                "val": 0.882,
                "test": 0.518,
                "wiou": 0.326
            },
            "permuted": {
                "id_val": 0.679,
                "val": 0.503,
                "test": 0.395,
            },
        },
    },

    "Motif size": {
        "ERM": {
            "original": {
                "test": 0.569,
            }
        },
        "LECI": {
            "original": {
                "test": 0.603,
                "wiou": 0.434
            },
            "permuted": {
                "test": 0.391,
            },
        },
        "CIGA": {
            "original": {
                "test": 0.497,
                "wiou": 0.555
            },
            "permuted": {
                "test": 0.436,
            },
        },
        "GSAT": {
            "original": {
                "test": 0.563,
                "wiou": 0.373
            },
            "permuted": {
                "test": 0.400,
            },
        },
    },

    "Motif2 basis": {
        "ERM": {
            "original": {
                "test": 0.406,
            }
        },
        "LECI": {
            "original": {
                "test": 0.852,
                "wiou": 0.769
            },
            "permuted": {
                "test": 0.391,
            },
        },
        "CIGA": {
            "original": {
                "test": 0.452,
                "wiou": 0.612
            },
            "permuted": {
                "test": 0.436,
            },
        },
        "GSAT": {
            "original": {
                "test": 0.763,
                "wiou": 0.329
            },
            "permuted": {
                "test": 0.400,
            },
        },
    },

    "GOODHIV": {
        "ERM": {
            "original": {
                "test": 0.697,
            }
        },
        "LECI": {
            "original": {
                "test": 0.712,
            },
            "permuted": {
                "test": 0.712,
            },
        },
        "CIGA": {
            "original": {
                "test": 0.652,
            },
            "permuted": {
                "test": 0.652,
            },
        },
        "GSAT": {
            "original": {
                "test": 0.734,
            },
            "permuted": {
                "test": 0.734,
            },
        },
    },

    "LBAPcore": {
        "ERM": {
            "original": {
                "test": 0.696,
            }
        },
        "LECI": {
            "original": {
                "test": 0.718,
            },
            "permuted": {
                "test": 0.718,
            },
        },
        "CIGA": {
            "original": {
                "test": 0.697,
            },
            "permuted": {
                "test": 0.696,
            },
        },
        "GSAT": {
            "original": {
                "test": 0.704,
            },
            "permuted": {
                "test": 0.704,
            },
        },
    }
}
print(data.keys())

markers = {
    "Motif basis": "o",
    "Motif2 basis": "*",
    "Motif size": "^",
    "GOODHIV": "s",
    "LBAPcore": "p"
}
colors = {
    "LECI": "blue",
    "CIGA": "orange", 
    "GSAT": "green"
}


split = "test"
max_acc_gain , min_acc_gain = 0, 0
for dataset in data.keys():
    for model in ["LECI", "CIGA", "GSAT"]:
        faith = abs(data[dataset][model]["permuted"][split] - data[dataset][model]["original"][split])
        acc_gain = data[dataset][model]["original"][split] - data[dataset]["ERM"]["original"][split]
        max_acc_gain , min_acc_gain = max(max_acc_gain, acc_gain), min(min_acc_gain, acc_gain)
        plt.scatter(faith, acc_gain, marker=markers[dataset], label=model, c=[colors[model]])

legend_elements = []
for key, value in markers.items():
    legend_elements.append(
        Line2D([0], [0], marker=value, color='w', label=key, markerfacecolor='grey', markersize=15)
    )
for key, value in colors.items():
    legend_elements.append(
        Patch(facecolor=value, label=key)
    )

plt.ylabel("Acc. Gain wrt ERM")
plt.xlabel("Faithfulness of Classifier")
plt.legend(handles=legend_elements) #, loc='center'
plt.savefig("GOOD/kernel/pipelines/plots/illustrations/scatter.png")
plt.close()


for dataset in ["Motif basis", "Motif2 basis", "Motif size"]:
    for model in ["LECI", "CIGA", "GSAT"]:
        faith = abs(data[dataset][model]["permuted"][split] - data[dataset][model]["original"][split])
        plaus = data[dataset][model]["original"]["wiou"]
        acc_gain = data[dataset][model]["original"][split] - data[dataset]["ERM"]["original"][split]
        plt.scatter(faith, plaus, marker=markers[dataset], label=model, c=acc_gain, vmin=min_acc_gain, vmax=max_acc_gain)

legend_elements = []
for key in ["Motif basis", "Motif2 basis", "Motif size"]:
    value = markers[key]
    legend_elements.append(
        Line2D([0], [0], marker=value, color='w', label=key, markerfacecolor='grey', markersize=15)
    )

plt.ylabel("Plausibility")
plt.xlabel("Faithfulness of Classifier")
plt.legend(handles=legend_elements)
plt.xlim(0,1)
plt.ylim(0,1)
plt.colorbar();
plt.savefig("GOOD/kernel/pipelines/plots/illustrations/scatter_plaus.png")