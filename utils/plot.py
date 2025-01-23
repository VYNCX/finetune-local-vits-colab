import logging
import matplotlib

matplotlib.use("Agg")

MATPLOTLIB_FLAG = False


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)  # Use ARGB format
    width, height = fig.canvas.get_width_height()
    data = data.reshape(height, width, 4)  # ARGB has 4 channels
    data = data[:, :, [1, 2, 3]]  # Reorder to RGB by dropping the alpha channel
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)  # Use ARGB format
    width, height = fig.canvas.get_width_height()
    data = data.reshape(height, width, 4)  # ARGB has 4 channels
    data = data[:, :, [1, 2, 3]]  # Reorder to RGB by dropping the alpha channel
    plt.close()
    return data
