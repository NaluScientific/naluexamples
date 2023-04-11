import matplotlib

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