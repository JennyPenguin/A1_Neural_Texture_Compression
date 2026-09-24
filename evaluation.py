import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

all_points = [
    ("gradient", "S3TC", 6.0, 131072, 38.26092956),
    ("gradient", "S", 15.4, 50956, 54.26163101),
    ("gradient", "M", 12.6, 62220, 57.26386642),
    ("gradient", "L", 2.1, 369932, 56.44772339),
    ("gradient", "S-q", 29.8, 26380, 53.57282257),
    ("gradient", "M-q", 26.2, 29964, 56.6919899),
    ("gradient", "L-q", 7.2, 108812, 56.27243042),

    ("bricks", "S3TC", 6.0, 131072, 36.62231189),
    ("bricks", "S", 15.4, 50956, 33.1704559),
    ("bricks", "M", 12.6, 62220, 35.9344101),
    ("bricks", "L", 2.1, 369932, 41.2156105),
    ("bricks", "S-q", 29.8, 26380, 32.6396713),
    ("bricks", "M-q", 26.2, 29964, 35.63796616),
    ("bricks", "L-q", 7.2, 108812, 41.2022857),

    ("clouds", "S3TC", 6.0, 131072, 37.50078965),
    ("clouds", "S", 15.4, 50956, 39.20388031),
    ("clouds", "M", 12.6, 62220, 41.57991791),
    ("clouds", "L", 2.1, 369932, 48.7183570),
    ("clouds", "S-q", 29.8, 26380, 39.09284973),
    ("clouds", "M-q", 26.2, 29964, 41.50522232),
    ("clouds", "L-q", 7.2, 108812, 48.6526641),

    ("car", "S3TC", 6.0, 131072, 28.77932968),
    ("car", "S", 15.4, 50956, 23.92092896),
    ("car", "M", 12.6, 62220, 26.0394592),
    ("car", "L", 2.1, 369932, 35.3689537),
    ("car", "S-q", 29.8, 26380, 23.0117130),
    ("car", "M-q", 26.2, 29964, 25.8104038),
    ("car", "L-q", 7.2, 108812, 35.183086),

    ("moon", "S3TC", 6.0, 131072, 28.7939358),
    ("moon", "S", 14.1, 50956, 26.59808922),
    ("moon", "M", 11.6, 62220, 27.19305038),
    ("moon", "L", 1.9, 369932, 29.1621589),
    ("moon", "S-q", 27.0, 26380, 26.58360481),
    ("moon", "M-q", 23.8, 29964, 27.18538094),
    ("moon", "L-q", 6.6, 108812, 29.1550293),

    ("night", "S3TC", 6.0, 131072, 20.6324802),
    ("night", "S", 7.9, 50956, 18.94382477),
    ("night", "M", 6.5, 62220, 20.67115784),
    ("night", "L", 1.1, 369932, 28.81870651),
    ("night", "S-q", 15.4, 26380, 18.86818314),
    ("night", "M-q", 13.5, 29964, 20.58025742),
    ("night", "L-q", 3.7, 108812, 28.61854172),
]

colors = {"gradient": "blue", "bricks": "red", "clouds": "purple",
          "car": "magenta", "moon": "green", "night": "pink"}

shapes = {"S": "v", "M": "<", "L": "^", 
          "S-q": "1", "M-q": "3", "L-q": "2", "S3TC": "P"}

for i in range(2):

    fig, ax = plt.subplots(figsize=(12, 8))

    size_offset = 500 if i == 0 else 0.5
    size_name = "bytes" if i == 0 else "factor"
        
    for image, size, compression_factor, raw_bytes, PSNR in all_points:
        size_axis = raw_bytes if i == 0 else compression_factor
        ax.scatter(size_axis, PSNR, color=colors[image], marker=shapes[size], s=150,)
        ax.text(size_axis+size_offset, PSNR+0.5, size)

    legend_elements = [
        Line2D([0], [0], color=c, label=image) for image, c in colors.items()
    ]

    legend_elements.extend([
        Line2D([0], [0], marker=marker, label=size, lw=0) for size, marker in shapes.items()
    ])

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.title("PSNR vs Size")
    if i == 0:
        plt.xlabel("Size (Bytes)")
    else:
        plt.xlabel("Size (Compression Factor)")
    plt.ylabel("PSNR (db)")
    plt.tight_layout()
    plt.savefig(f'Evaluation_{size_name}.png', dpi=100)
    plt.clf()

