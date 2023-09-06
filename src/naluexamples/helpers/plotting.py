import matplotlib
import matplotlib.pyplot as plt


def set_plot_style(font_size: int = 18, font_family: str = "monospace"):
    """Sets the matplotlib font size and family
    """
    matplotlib.rcParams.update({"font.size": font_size})
    matplotlib.rcParams.update({"font.family": font_family})

def get_color_mapping(name: str = "ocean"):
    """Gets a matplot lib color map
    """
    cmap = matplotlib.cm.get_cmap(name)
    return cmap


def simple_event_plot(event: dict, title: "str|None" = None, xlabel: "str|None" = None, ylabel: "str|None" = None, legend: bool=True, cmap = None):
    """Plot a single event in a simple line graph.

    Args:
        event (dict): the event to plot. Must be parsed.
        title (str): the title of the plot.
        legend (bool): whether to show the legend.
        cmap: the color map to use for the lines.
    """
    plt.title(title or f"Event {event['event_num']}")
    plt.xlabel(xlabel or "Sample Number")
    plt.ylabel(ylabel or "ADC Value (counts)")
    for channel, data in enumerate(event["data"]):
        extra_kwargs = {}
        if cmap:
            extra_kwargs["color"] = cmap(channel / len(event["data"]))
        plt.plot(data, label=f"Channel {channel}", **extra_kwargs)
    if legend:
        plt.legend()
    plt.show()
