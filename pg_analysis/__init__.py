import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif',
                     'figure.figsize': [5.5, 2.8],
                     'figure.dpi': 120,
                     # 'figure.frameon': False,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16,
                     'axes.labelsize': 16,
                     'axes.titlesize': 18,
                     'legend.fontsize': 15,
                     'legend.frameon': False,
                     'legend.handlelength': 1.0,
                     'legend.handletextpad': 0.5,
                     'legend.labelspacing': 0.5,
                     })

# print(plt.rcParams.keys())

COLORS = ['#49b8ff', '#ff7575', '#66c56c', '#f4b247', "#5c5c5c"]
LINE_STYLES = ['-', '-.', '--', ':', '.']

from matplotlib.ticker import FuncFormatter


def time_fmt(time):
    seconds = int(time)
    yrs = seconds // 31_557_600
    wks = seconds % 31_557_600 // 2_629_800
    dys = seconds % 2_629_800 // 86400
    hrs = seconds % 86_400 // 3600
    mns = seconds % 3600 // 60
    scs = seconds % 60
    return yrs, wks, dys, hrs, mns, scs


def set_timedelta_formatter(x_data):
    try:
        data_min, data_max = plt.gca().get_xlim()
    except:
        data_min, data_max = x_data.min(), x_data.max()

    # yrs, wks, dys, hrs, mns, scs = time_fmt(data_min)
    offset = 0 if data_min < data_max // 2 else data_min

    def timedelta_formatter(seconds, pos):
        yrs, wks, dys, hrs, mns, scs = time_fmt(seconds - offset)

        if yrs:
            return f"{yrs}y"
        elif wks:
            return f"{wks}w"
        elif dys:
            return f"{dys}d"
        elif hrs:
            return f"{hrs}h"
        elif mns:
            return f"{mns}m"
        else:
            return f"{scs}"

    plt.gca().xaxis.set_major_formatter(FuncFormatter(timedelta_formatter))

    if offset:
        plt.xlim(offset, data_max)


def get_dian(x):
    scale = int(10 ** 20)
    data_dian = 0
    while not data_dian:
        scale /= 10
        data_dian = x // scale

    return data_dian * scale, scale


def set_scientific_formatter(data, offset=None, scale=1, axis="x"):
    # todo: replace with xlim values
    try:
        if axis == "x":
            data_min, data_max = plt.gca().get_xlim()
        else:
            data_min, data_max = plt.gca().get_ylim()
    except:
        data_min, data_max = data.min(), data.max()

    if offset is None:
        offset = 0

        if data_min > (data_max - data_min):
            offset, new_scale = get_dian(data_min)

    scale_str = ""
    # if scale is None:
    #     _, scale = get_dian(int(data_max - offset) // 50)
    if scale == 'k':
        scale_str = scale
        scale = 1000
    elif scale == 'M':
        scale_str = scale
        scale = 1000_000
    elif scale == 'b':
        scale_str = scale
        scale = 1000_000_000
    elif scale == 'T':
        scale_str = scale
        scale = 1000_000_000_000

    def scalar_formatter(x, pos):
        # if pos == 4:
        #     return f"{int((x - offset) / scale)}$\\times$10$^{{{int(math.log(scale, 10))}}}$"
        value = int((x - offset) / scale)
        return f"{value}{scale_str if value else ''}"

    if axis == "x":
        plt.gca().xaxis.set_major_formatter(FuncFormatter(scalar_formatter))
    else:
        plt.gca().yaxis.set_major_formatter(FuncFormatter(scalar_formatter))

    if offset:
        if axis == "x":
            plt.xlim(offset, data_max)
        else:
            plt.ylim(offset, data_max)


def plot_area(df, xKey, *yKeys, k=40, label=None, labels=None, color=None, colors=None,
              x_format="scalar", x_opts={}, y_format="scalar", y_opts={}):
    labels = [label] if label else (labels or yKeys)
    colors = [color] if color else (colors or COLORS)

    for i, (yKey, label) in enumerate(zip(yKeys, labels)):
        series = df[[xKey, yKey]].dropna()
        if k is not None:
            series['binned'] = pd.qcut(series[xKey], q=k, precision=0)
            grouped = series.groupby('binned')
            xs = grouped[xKey].min().reset_index()[xKey]
            ys = grouped[yKey].mean().reset_index()[yKey]
            std = grouped[yKey].std().reset_index()[yKey]
        else:
            grouped = series.groupby(xKey).mean().reset_index()
            xs = grouped[xKey]
            ys = grouped[yKey]
            std = series.groupby(xKey).std().reset_index()[yKey]

        plt.plot(xs, ys, color=colors[i % len(colors)], label=label)
        plt.fill_between(xs, ys - std, ys + std,
                         color=colors[i % len(colors)], alpha=0.4, linewidth=0)

    if x_format == "scalar":
        set_scientific_formatter(xs, **x_opts)
    elif x_format == "timedelta":
        set_timedelta_formatter(xs, **x_opts)
    if y_format == "scalar":
        set_scientific_formatter(xs, **y_opts, axis="y")
    elif y_format == "timedelta":
        set_timedelta_formatter(xs, **y_opts)

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
