import streamlit as st

from scipy.interpolate import interp1d, CubicSpline

from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from sympy.abc import x

import numpy as np
import pandas as pd

import altair as alt

import matplotlib.pyplot as plt
import matplotlib.font_manager


# make sure the humor sans font is found. This only needs to be done once
# on a system, but it is done here at start up for usage on share.streamlit.io.
matplotlib.font_manager.findfont('Humor Sans', rebuild_if_missing=True)

# Define some helpful functions

# we need helper functions to interactively update horizontal and vertical lines in a plot
# https://stackoverflow.com/questions/29331401/updating-pyplot-vlines-in-interactive-plot

def update_vlines(*, h, x, ymin=None, ymax=None):
    """
    If h is a handle to a vline object in a matplotlib plot, this function can be used to update x, ymin, ymax
    """
    seg_old = h.get_segments()
    if ymin is None:
        ymin = seg_old[0][0, 1]
    if ymax is None:
        ymax = seg_old[0][1, 1]

    seg_new = [np.array([[x, ymin],
                         [x, ymax]]), ]

    h.set_segments(seg_new)


def update_hlines(*, h, y, xmin=None, xmax=None):
    """
    If h is a handle to a hline object in a matplotlib plot, this function can be used to update y, xmin, xmax
    """
    seg_old = h.get_segments()
    if xmin is None:
        xmin = seg_old[0][0, 0]
    if xmax is None:
        xmax = seg_old[0][1, 0]

    seg_new = [np.array([[xmin, y],
                         [xmax, y]]), ]

    h.set_segments(seg_new)


#############################################
# Define the function that updates the plot #
#############################################

#@st.cache(suppress_st_warning=True)
def update_data(interptype,t0,ti_input,yi_input,resolution):
    """
    y_interp,y = update_data(interptype,t,ti,yi)
    """
    
    
    ti = string_to_list(ti_input)
    yi = string_to_list(yi_input)
    
    tmin = min(ti)
    tmax = max(ti)
    length = max(ti)-min(ti)
    dt = length/resolution
    
    t_interp = np.arange(tmin,tmax,dt)
    if interptype == 'linear':
        y_interp=interp1d(ti,yi)
    elif interptype == 'spline':
        y_interp=CubicSpline(ti,yi)
    ft0 = float(y_interp(t0))
    y_interp = y_interp(t_interp)
    
    return t_interp,y_interp,ft0,ti,yi


def string_to_list(stringlist):
    list_of_str = stringlist.split()
    list_from_str = [float(x) for x in list_of_str]
    return list_from_str

# To Do: Why does caching update_plot hang?
# @st.cache(suppress_st_warning=True)
def update_plot(ti, yi, t0, ft0, t_interp, y_interp, visible, ti_input, yi_input):
    
    """
    Creates a Matplotlib plot if the dictionary st.session_state.handles is empty, otherwise
    updates a Matplotlib plot by modifying the plot handles stored in st.session_state.handles.
    The figure is stored in st.session_state.fig.

    :param x0: Evaluation point of the function/Taylor polynomial
    :param fx0: Function evaluated at x0
    :param xs: numpy-array of x-coordinates
    :param ys: numpy-array of f(x)-coordinates
    :param ps: numpy-array of P(x)-coordinates, where P is the Taylor polynomial
    :param visible: A flag wether the Taylor polynomial is visible or not
    :param xmin: minimum x-range value
    :param xmax: maximum x-range value
    :param ymin: minimum y-range value
    :param ymax: maximum y-range value
    :return: none.
    """
    
    if type(ti) == 'str':
        ti = string_to_list(ti_input)
    if type(yi) == 'str':
        yi = string_to_list(yi_input)
    
    tmin = min(ti)
    tmax = max(ti)
    ymin = min(min(yi),min(y_interp))
    ymax = max(max(yi),max(y_interp))

    handles = st.session_state.handles

    ax = st.session_state.mpl_fig.axes[0]

    # if the dictionary of plot handles is empty, the plot does not exist yet. We create it. Otherwise the plot exists,
    # and we can update the plot handles in fs, without having to redraw everything (better performance).
    if not handles:
        #######################
        # Initialize the plot #
        #######################

        # plot the Taylor polynomial
        handles["datapoints"] = ax.plot(ti, yi,
                                        color='g',
                                        linewidth=0,
                                        marker='o',
                                        ms=15,
                                        label='Data points'.format(degree))[0]

        # plot f and append the plot handle
        handles["interpol"] = ax.plot(t_interp, y_interp,
                                      color='b',
                                      label="Interpolation of data points")[0]

        handles["interpol"].set_visible(visible)

        ###############################
        # Beautify the plot some more #
        ###############################

        plt.title('Interpolation of a series of data points')
        plt.xlabel('t', horizontalalignment='right', x=1)
        plt.ylabel('y', horizontalalignment='right', x=0, y=1)

        # set the z order of the axes spines
        for k, spine in ax.spines.items():
            spine.set_zorder(0)

        # set the axes locations and style
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['right'].set_color('none')

        # draw lines for (x0, f(x0))
        handles["vline"] = plt.vlines(x=t0, ymin=float(min(0, ft0)), ymax=float(max(0, ft0)), colors='black', ls=':', lw=2)
        handles["hline"] = plt.hlines(y=float(ft0), xmin=tmin, xmax=t0, colors='black', ls=':', lw=2)

    else:
        ###################
        # Update the plot #
        ###################

        # Update the data points plot
        handles["datapoints"].set_xdata(ti)
        handles["datapoints"].set_ydata(yi)

        # update the interpolation plot
        handles["interpol"].set_xdata(t_interp)
        handles["interpol"].set_ydata(y_interp)

        # update the visibility of the Taylor expansion
        handles["interpol"].set_visible(visible)

        update_vlines(h=handles["vline"], x=t0, ymin=float(min(0, ft0)), ymax=float(max(0, ft0)))
        update_hlines(h=handles["hline"], y=float(ft0), xmin=tmin, xmax=t0)

    # set x and y ticks, labels and limits respectively
    xticks = []
    xticklabels = []
    if tmin <= 0 <= tmax:
        xticks.append(0)
        xticklabels.append("0")
    if tmin <= t0 <= tmax:
        xticks.append(t0)
        xticklabels.append("x0")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    yticks = []
    yticklabels = []
    if ymin <= 0 <= ymax:
        yticks.append(0)
        yticklabels.append("t0")
    if ymin <= ft0 <= ymax:
        yticks.append(ft0)
        yticklabels.append("f_interp(t0)")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # set the x and y limits
    ax.set_xlim([tmin, tmax])
    ax.set_ylim([ymin, ymax])

    # show legend
    legend_handles = [handles["datapoints"], ]
    if visible:
        legend_handles.append(handles["interpol"])
    ax.legend(handles=legend_handles,
              loc='lower center',
              bbox_to_anchor=(0.5, -0.15),
              ncol=2)

    # make all changes visible
    st.session_state.mpl_fig.canvas.draw()

if __name__ == '__main__':

    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

    # create sidebar widgets

    st.sidebar.title("Advanced settings")

    #func_str = st.sidebar.text_input(label="function",
    #                                 value='25 + exp(x)*sin(x**2) - 10*x')

    st.sidebar.markdown("Visualization Options")

    # Good for in-classroom use
    qr = st.sidebar.checkbox(label="Display QR Code", value=False)

    toggle_interp = st.sidebar.checkbox(label='Display Interpolation', value=True)

    # degree_max = 10
    # if toggle_interp:
    #     degree_max = st.sidebar.number_input(label='max degree', value=10)

    # xcol1, xcol2 = st.sidebar.columns(2)
    # with xcol1:
    #     tmin = st.number_input(label='tmin', value=1.)
    #     ymin = st.number_input(label='ymin', value=-50.)
    # with xcol2:
    #     tmax = st.number_input(label='tmax', value=4.)
    #     ymax = st.number_input(label='ymax', value=50.)

    
    res = st.sidebar.number_input(label='resolution', value=100, step=10)

    backend = 'Matplotlib' #st.sidebar.selectbox(label="Backend", options=('Matplotlib', 'Altair'), index=0)

    # Create main page widgets

    tcol1, tcol2 = st.columns(2)

    with tcol1:
        st.title('Interpolated Data Points')
    with tcol2:
        if qr:
            st.markdown('## <img src="https://api.qrserver.com/v1/create-qr-code/?size=150x150&data='
                        'https://share.streamlit.io/PhiSpel/spielbeispiel-interpolation/main" width="200"/>',
                        unsafe_allow_html=True)

    # prepare matplotlib plot
    if 'Matplotlib' in backend:

        def clear_figure():
            del st.session_state['mpl_fig']
            del st.session_state['handles']
        xkcd = st.sidebar.checkbox("use xkcd-style", value=True, on_change=clear_figure)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        interptype = st.selectbox(label="interpolation type", options=('linear', 'spline'), index=0)
        
    with col2:
        ti_input = st.text_input(label='time values, space-separated, same amount as sensor values!',
                                 value="0 1 2 3 4",
                                 placeholder="please input time values")
    with col3:
        yi_input = st.text_input(label='sensor values, space-separated, same amount as time values!',
                                 value="0 1 4 1 0",
                                 placeholder="please input sensor values")
    
    with col4:
        t0 = st.slider(
                't0',
                min_value=float(0),
                max_value=float(10),
                value=float(1.5)
            )
    
    # update the data
    #coefficients = update_coefficients(func_str, degree_max)
    t_interp,y_interp,ft0,ti,yi = update_data(interptype,t0,ti_input,yi_input,res)
    #xs, ys, ps, fx0 = update_data(coefficients, degree, x0, xmin, xmax, res)

    if 'Matplotlib' in backend:

        if xkcd:
            # set rc parameters to xkcd style
            plt.xkcd()
        else:
            # reset rc parameters to default
            plt.rcdefaults()

        # initialize the Matplotlib figure and initialize an empty dict of plot handles
        if 'mpl_fig' not in st.session_state:
            st.session_state.mpl_fig = plt.figure(figsize=(8, 3))
            st.session_state.mpl_fig.add_axes([0., 0., 1., 1.])

        if 'handles' not in st.session_state:
            st.session_state.handles = {}

    if 'Altair' in backend and 'chart' not in st.session_state:
        # initialize empty chart
        st.session_state.chart = st.empty()

    # update plot
    if 'Matplotlib' in backend:
        update_plot(ti, yi, t0, ft0, t_interp, y_interp, toggle_interp,ti_input,yi_input)
        st.pyplot(st.session_state.mpl_fig)
    else:
        df = pd.DataFrame(data=np.array([ti, yi, t_interp, y_interp], dtype=np.float64).transpose(),
                          columns=["ti", "yi", "t_range", "interpolation"])
        chart = alt.Chart(df) \
            .transform_fold(["yi", "interpolation"], as_=["legend", "y"]) \
            .mark_line(clip=True) \
            .encode(
                x=alt.X('x:Q', scale=alt.Scale(domain=(min(ti), max(ti)))),
                y=alt.Y('y:Q', scale=alt.Scale(domain=(min(yi), max(yi)))),
                color=alt.Color('legend:N',
                                scale=alt.Scale(range=["green", "blue"]),
                                legend=alt.Legend(orient='bottom'))
            )\
            .interactive()
        pnt_data = pd.DataFrame({'x': [float(t0),], 'y': [float(ft0),]})
        pnt = alt.Chart(pnt_data)\
            .mark_point(clip=True, color='white')\
            .encode(
                x='x:Q',
                y='y:Q',
            )\
            .interactive()
        altair_chart = (chart + pnt).properties(width=800, height=400)
        st.session_state.chart.altair_chart(altair_chart, use_container_width=True)
