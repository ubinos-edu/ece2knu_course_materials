import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import io
import os
from scipy.signal import savgol_filter # 사비츠키-골레이(Savitzky-Golay) 필터
from scipy import interpolate
from numpy import nan, inf

from IPython.display import Image

# %matplotlib tk
# %matplotlib inline

def tangent_line(f, x):
    h = 1e-4
    d = (f(x+h) - f(x-h)) / (2*h)
    return lambda t: d*t - d*x + f(x)

def tangent_line_end(f, x):
    h = 1e-4
    d = (f(x) - f(x-h)) / (h)
    return lambda t: d*t - d*x + f(x)

def tangent_line_and_gradient(f, x):
    h = 1e-4
    d = (f(x+h) - f(x-h)) / (2*h)
    return (lambda t: d*t - d*x + f(x)), d

def tangent_line_and_gradient_end(f, x):
    h = 1e-4
    d = (f(x) - f(x-h)) / (h)
    return (lambda t: d*t - d*x + f(x)), d

def derivative(f, x):
    h = 1e-4
    d = (f(x+h) - f(x-h)) / (2*h)
    return d

def get_simulation_result(file_name, start=0, end=-1):
    if not os.path.exists(file_name):
        print("{} doesn't exist.".format(file_name))
        return None

    if end == -1:
        end = None
    else:
        end += 1

    with open(file_name, encoding='cp1252') as data_file:
        lines = data_file.read()
        occurrences = lines.count('Step Information:')
        data_file.seek(0)

        line = data_file.readline()
        labels = re.split(', | ,|\t', line)
        labels = [s.strip().upper() for s in labels]

        data = {}

        if (occurrences == 0):
            for label in labels:
                data[label] = []

            for line in data_file:
                values = re.split(', | ,|\t', line)
                for i in range(len(values)):
                    value = float(values[i]) * 1000
                    data[labels[i]].append(value)

        else:
            labels_all =[]

            lines = data_file.readline() # skip first line starting with 'Step Information:'

            for idx in range(occurrences):
                labels_new = []
                for label in labels:
                    labels_new.append('(%s)@%d' % (label, idx+1))
                labels_all += labels_new

                for label_new in labels_new:
                    data[label_new] = []

                for line in data_file:
                    if (line.startswith('Step Information:')):
                        break
                    values = re.split(', | ,|\t', line)
                    for i in range(len(values)):
                        value = float(values[i]) * 1000
                        data[labels_new[i]].append(value)

            labels = labels_all

        for label in labels:
            data[label] = np.array(data[label][start:end])

    # print("labels = ", end='')
    # print(list(data.keys()))
    for label in list(data.keys()):
        print("data['%s'] : sample number = %d" % (label, len(data[label])))

    return data

def get_oscilloscpoe_result_tektronix(file_name, start=0, end=-1):
    if not os.path.exists(file_name):
        print("{} doesn't exist.".format(file_name))
        return None

    if end == -1:
        end = None
    else:
        end += 1

    df = pd.read_csv(file_name, header=None, encoding='cp1252',low_memory=False)

    label_ri = df.loc[df.iloc[:,0] == 'Source', 1].index
    ci = 0
    data = {}
    data['TIME'] = df.iloc[start:end, 3].to_numpy().astype(float)
    data['TIME'] -= data['TIME'][0]
    for idx in range(df.shape[1]//6):
        label = df.iloc[label_ri, ci+1].item()
        data[label] = df.iloc[start:end,ci+4].to_numpy().astype(float)
        ci += 6

    # print("labels = ", end='')
    # print(list(data.keys()))
    for label in list(data.keys()):
        print("data['%s'] : sample number = %d" % (label, len(data[label])))

    return data

def print_array(label, values):
    print('%s = [' % label, end='')
    for idx, vd in enumerate(values):
        print('{:11.3f}'.format(vd), end='')
        if (idx+1 != len(values)):
            print(', ', end='')
    print(']')

def print_value(label, value):
    print('%s = ' % label, end='')
    print('{:11.3f}'.format(value))

def print_value_to_string(label, value):
    output = io.StringIO()
    print('%s = ' % label, end='', file=output)
    print('{:11.3f}'.format(value), file=output)
    captured = output.getvalue()
    return captured

def draw_plot(xs, ys, label, style_idx, color_idx=-1, marker_num=16, scatter=False, scatter_s=2):
    linestyle  = ['-',          '-',            '-',            '-',            '-',
                  '-',          '-',            '-',            '-',            '-',
                  '-',          '-',            '-',            '-',            '-',
                  '-',          '-',            '-',            '-',            '-',
                  '-',          '-',            '-',            '-',            '-']
    colors     = ['blue',       'green',        'red',          'orange',       'purple',
                  'cyan',       'darkseagreen', 'brown',        'goldenrod',    'darkviolet',
                  'steelblue',  'limegreen',    'tomato',       'tan',          'deeppink',
                  'navy',       'lightgreen',   'indianred',    'khaki',        'rebeccapurple',
                  'slategray',  'forestgreen',  'orangered',    'wheat',        'orchid']
    markers    = ['o',          'v',            '<',            's',            'p',
                  'h',          '*',            'X',            'x',            '^',
                  '>',          'P',            'D',            'H',            'd',
                  '|',          '-',            '4'             '5',            '6',
                  '7',          '8',            '9 ',           '10',           '+']

    if color_idx < 0:
        color_idx = style_idx

    if isinstance(xs, list):
        xs = np.array(xs)
    if isinstance(ys, list):
        ys = np.array(ys)

    if (marker_num == 0):
        selected_markevery = None
        selected_marker = None
    elif not isinstance(xs, np.ndarray):
        selected_markevery = None
        selected_marker = markers[style_idx]
    elif (marker_num > 0):
        markeverys = []
        if xs[0] > xs[-1]:
            first_idx = -1
        else:
            first_idx = 0
        x_step = (xs.max() - xs.min()) / marker_num
        for idx in range(len(markers)):
            x_start = xs.min() + ((x_step / (len(markers) + 1)) * idx)
            marker_list = []
            for j in range(marker_num):
                tmp_array = np.where(xs > (x_start + (x_step * j)))
                if (len(tmp_array[0]) > 0):
                    marker_list.append(tmp_array[0][first_idx])
            markeverys.append(marker_list)

        selected_markevery = markeverys[style_idx]
        selected_marker = markers[style_idx]
    else:
        selected_markevery = None
        selected_marker = markers[style_idx]

    if (scatter):
        plt.scatter(xs, ys, color=colors[color_idx], marker=selected_marker, s=scatter_s, label=label)
    else:
        plt.plot(xs, ys, ls=linestyle[style_idx], color=colors[color_idx], marker=selected_marker, markevery=selected_markevery, label=label)
    if (label != None):
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

def set_plot_size(xrate, yrate):
    if (len(set_plot_size.g_plt_figsize) == 0):
        set_plot_size.g_plt_figsize = plt.rcParams["figure.figsize"]
    size = []
    size.append(set_plot_size.g_plt_figsize[0] * xrate)
    size.append(set_plot_size.g_plt_figsize[1] * yrate)
    plt.figure(figsize=size)

set_plot_size.g_plt_figsize = []


def calculate_square_wave_frequency(times, vins):
    # 전압 값의 중간값을 임계값으로 설정
    threshold = (np.max(vins) + np.min(vins)) / 2

    # 상승 에지 감지
    rising_edges = np.where((vins[:-1] < threshold) & (vins[1:] >= threshold))[0]

    if len(rising_edges) < 2:
        return None  # 주기를 계산할 수 없음

    # 주기 계산 (연속된 상승 에지 사이의 평균 시간)
    periods = times[rising_edges[1:]] - times[rising_edges[:-1]]
    average_period = np.mean(periods)

    # 주파수 계산 (주기의 역수)
    frequency = 1 / average_period

    return frequency

def display_image(file_path, width=None, height=None):
    if os.path.exists(file_path):
        display(Image(file_path, width=width, height=height))
    else:
        print("{} doesn't exist.".format(file_path))

def find_first_peak_index(xs, ys, margine=10):
    xs_len = len(xs)
    old_grad = None
    pi = -1
    m = margine
    for i in range(xs_len - 1):
        grad = (ys[i+1] - ys[i]) / (xs[i+1] - xs[i])
        if grad > 0:
            pi = i
            m -= 1
            if m == 0:
                old_grad = grad
                break
        else:
            m = margine
    if old_grad != None:
        m = margine
        for i in range(pi, xs_len - 1):
            grad = (ys[i+1] - ys[i]) / (xs[i+1] - xs[i])
            if grad < 0:
                pi = i
                m -= 1
                if m == 0:
                    pi -= margine
                    break
            else:
                m = margine
    return pi
