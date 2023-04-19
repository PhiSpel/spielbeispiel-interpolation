import streamlit as st

from scipy.interpolate import interp1d, CubicSpline

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
def update_data(interptype, t0, ti_input, yi_input, resolution, degree):
    """
    y_interp,y = update_data(interptype,t,ti,yi)
    """
    
    
    ti = string_to_list(ti_input)
    yi = string_to_list(yi_input)
    
    tmin = min(ti)
    tmax = max(ti)
    length = max(ti)-min(ti)
    dt = length/resolution
    
    t_interp = np.arange(tmin, tmax, dt)
    if interptype == 'linear':
        y_interp = interp1d(ti, yi)
        # reverse-engineering function near t0
        dtlin = dt/100
        yright = y_interp(t0+dtlin)
        yleft = y_interp(t0-dtlin)
        factor_lin = (yright - yleft)/ (2*dtlin)
        factor_const = float(y_interp(t0)) - factor_lin*t0
        factors=[factor_lin,factor_const]
    elif interptype == 'spline':
        y_interp=CubicSpline(ti, yi)
        i = np.searchsorted(ti, t0)-1
        factors=[y_interp.c[0, i], y_interp.c[1, i], y_interp.c[2, i], y_interp.c[3, i]]
        # convert factors from fac*(x-x[i])**(3-k) to fac*x**(3-k)
        # (x-xi)^3 = x^3 - 3xi x^2 + 3xi^2 x - xi^3
        # (x-xi)^2 =           x^2 - 2xi   x + xi^2
        # (x-xi)^1 =                       x - xi
        xi = ti[i]
        fac3 = factors[0]*1
        fac2 = factors[0]*-3*xi     + factors[1]*1
        fac1 = factors[0]*3*(xi**2) + factors[1]*-2*xi   + factors[2]*1
        fac0 = factors[0]*-(xi**3)+ factors[1]*(xi**2) + factors[2]*-xi + factors[3]*1
        factors = [fac3, fac2, fac1, fac0]
    elif interptype == 'polynomial':
        z = np.polyfit(ti, yi, degree)
        y_interp = np.poly1d(z)
        factors = z
    else:
        return ValueError
    ft0 = float(y_interp(t0))
    y_interp = y_interp(t_interp)
    
    return t_interp, y_interp, ft0, ti, yi, factors


def string_to_list(stringlist):
    list_of_str = stringlist.split()
    list_from_str = [float(x) for x in list_of_str]
    return list_from_str


def bmatrix(a, matrixtype = ''):
    if matrixtype == 'b':
        text = r' \begin{bmatrix} '
    elif matrixtype == 'v':
        text = r' \begin{matrix} '
    else:
        return ValueError
    #text += '\n'
    for x in range(len(a)):
        for y in range(len(a[x])):
            text += str(a[x][y])
            text += r' & '
        text = text[:-2]
        text += r'\\'
        text += '\n'
    if matrixtype == 'b':
        text += r' \end{bmatrix} '
    elif matrixtype == 'v':
        text += r' \end{matrix} '
    return text


def print_equations(ti,yi,factors,textsize):
    # preparations
    matrix = [['' for j in range(len(factors))] for j in range(len(ti))]
    for i in range(len(ti)):
        for j in range(len(factors)):
            if j == 0:
                string = '1'
            else:
                string = r'x_{' + str(i+1) + r'}'
            if j > 1:
                string += r'^{' + str(j) + r'}'
            matrix[i][j] = string
    # factors
    factors = np.round(factors,3)
    factors[factors==0] = 0
    factors = [[factors[i]] for i in range(factors.size-1,-1,-1)]
    # yi
    yi = np.round(yi,3)
    yi[yi==0] = 0
    yi = [[yi[i]] for i in range(yi.size)]
    
    # printing
    equation_string = r'$'
    equation_string += textsize
    equation_string += bmatrix(matrix,'b')
    equation_string += '\cdot '
    #st.write(factors)
    equation_string += ' ? '
    equation_string += ' = '
    yi_empty = [[''] for j in range(len(factors))]
    for j in range(len(factors)):
        yi_empty[j][0] = r'y_{' + str(j+1) + r'}'
    equation_string += bmatrix(yi_empty,'b')
    # 2nd equation
    equation_string += r'\rightarrow'
    matrix = [['' for j in range(len(factors))] for j in range(len(ti))]
    for i in range(len(ti)):
        for j in range(len(factors)):
            if j == 0:
                string = '1'
            else:
                string = str(ti[i])
            if j > 1:
                string += r'^{' + str(j) + r'}'
            matrix[i][j] = string
    equation_string += bmatrix(matrix,'b')
    equation_string += '\cdot '
    equation_string += ' ? '
    equation_string += '='
    equation_string += bmatrix(yi,'b')
    # 3rd equation
    equation_string += r'\rightarrow'
    matrix = [['' for j in range(len(factors))] for j in range(len(ti))]
    for i in range(len(ti)):
        for j in range(len(factors)):
            if j == 0:
                string = '1'
            else:
                string = str(ti[i]**j)
            matrix[i][j] = string
    equation_string += bmatrix(matrix,'b')
    equation_string += '\cdot '
    equation_string += bmatrix(factors,'b')
    equation_string += ' = '
    equation_string += bmatrix(yi,'b')
    equation_string += r'$'
    return equation_string


# To Do: Why does caching update_plot hang?
# @st.cache(suppress_st_warning=True)
def update_plot(ti, yi, t0, ft0, t_interp, y_interp, visible, ti_input, yi_input, ticks_on):
    
    """
    Creates a Matplotlib plot if the dictionary st.session_state.handles is empty, otherwise
    updates a Matplotlib plot by modifying the plot handles stored in st.session_state.handles.
    The figure is stored in st.session_state.fig.

    :param t0: Evaluation point of the function/Taylor polynomial
    :param ft0: Function evaluated at t0
    :param ti: numpy-array of x-coordinates
    :param yi: numpy-array of f(x)-coordinates
    :param y_interp: numpy-array of P(x)-coordinates, where P is the Taylor polynomial
    :param visible: A flag whether the Taylor polynomial is visible or not
    :return: none.
    """
    
    if type(ti) == 'str':
        ti = string_to_list(ti_input)
    if type(yi) == 'str':
        yi = string_to_list(yi_input)
    
    tmin = min(ti)
    tmax = max(ti)
    #if tmax-tmin > 5:
    #    dt = round((tmax-tmin)/10)
    #elif tmax-tmin >= 2:
    #    dt = 0.5
    #else:
    #    dt = 0.1
    #dt = 1

    ymin = min(min(yi), min(y_interp))
    ymax = max(max(yi), max(y_interp))
    height = ymax - ymin
    if height > 5:
        dy = round(height/10)
    elif height > 2:
        dy = 0.5
    else:
        dy = 0.1

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
                                        label='Data points')[0]#.format(degree))[0]

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

        # draw lines for (t0, f(t0))
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

    # set t and y ticks, labels and limits respectively
    if ticks_on:
        xticks = [x for x in ti]#range(round(tmin-0.5),round(tmax+0.5),dt)]
    else:
        xticks=[]
    xticklabels = [str(x) for x in xticks]
    
    if tmin <= 0 <= tmax:
        xticks.append(0)
        xticklabels.append("0")
    if tmin <= t0 <= tmax:
        xticks.append(t0)
        xlabel_string = "t\N{SUBSCRIPT ZERO}=" + str(round(t0,1))
        xticklabels.append(xlabel_string)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    if ticks_on:
        yticks = [x for x in yi]
    else:
        yticks=[]
    yticklabels = [str(x) for x in yticks]
    if ymin <= 0 <= ymax:
        yticks.append(0)
        yticklabels.append("0")
    if ymin <= ft0 <= ymax:
        yticks.append(ft0)
        ylabel_string = "f(t\N{SUBSCRIPT ZERO})=" + str(round(ft0,1))
        yticklabels.append(ylabel_string)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # set the x and y limits
    ax.set_xlim([tmin-0.5, tmax+0.5])
    ax.set_ylim([ymin-0.5, ymax+0.5])
    

    # show legend
    legend_handles = [handles["datapoints"], ]
    if visible:
        legend_handles.append(handles["interpol"])
    ax.legend(handles=legend_handles,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.15),
              ncol=2)

    # make all changes visible
    st.session_state.mpl_fig.canvas.draw()


def make_description(interptype, factors):
    if interptype == 'linear':
        factor_lin_round = round(factors[0],3)
        factor_const_round = round(factors[1],3)
        linear_description = r'''
        $f$ with linear approximation around $t_0$ is $\approx'''
        if not factor_lin_round == 0:
            linear_description+= str(factor_lin_round) + '''t'''
        if factor_const_round > 0:
            linear_description+='''+''' + str(factor_const_round) + '''$'''
        elif factor_const_round==0:
            linear_description+='''$'''
        else:
            linear_description+=str(factor_const_round) + '''$'''
        description=linear_description
    elif interptype == 'spline':
        factors = [round(elem,3) for elem in factors]
        spline_description = r'''
        $f$ with spline approximation around $t_0$ is $\approx'''
        for i in range(0,4,1):
            if (factors[i] > 0) & (i>0):
                spline_description += '+'
            if not factors[i] == 0:
                if (3-i) > 1:
                    spline_description += str(factors[i]) + 't^' + str(3-i)
                elif (3-i) == 1:
                    spline_description += str(factors[i]) + 't'
                else:
                    spline_description += str(factors[i])
        spline_description+= '''$'''
        description = spline_description
    elif interptype == 'polynomial':
        polynomial_description = r'''$f$ calculated by np.polyfit is $$f(t) \approx '''
        factors = [round(elem, 3) for elem in factors]
        deg = len(factors)
        for i in range(0, deg):
            if (factors[i] > 0) & (i < deg):
                polynomial_description += '+'
            degree = deg-i-1
            if not factors[i] == 0:
                if degree > 1:
                    polynomial_description += str(factors[i]) + 't^{' + str(degree) + '}'
                elif degree == 1:
                    polynomial_description += str(factors[i]) + 't'
                elif degree == 0:
                    polynomial_description += str(factors[i])
        polynomial_description += '''$$'''
        description = polynomial_description
    else:
        return ValueError
    return description


if __name__ == '__main__':
    debug = False
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

    # create sidebar widgets
    st.sidebar.title("Advanced settings")
    st.sidebar.markdown("Visualization Options")

    # Good for in-classroom use
    qr = st.sidebar.checkbox(label="Display QR Code", value=False)
    toggle_interp = st.sidebar.checkbox(label='Display Interpolation', value=True)
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
        xkcd = st.sidebar.checkbox("use xkcd-style", value=False, on_change=clear_figure)
    else:
        raise EnvironmentError

    ticks_on = st.sidebar.checkbox("show xticks and yticks", value=True, on_change=clear_figure)
    
    # prepare standard values input
    yi_std_str = "0 1 4 10 -3 3"
    ti_std_str = "0 1 2 3 4 5"
    ti_std = string_to_list(ti_std_str)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        interptype = st.selectbox(label="interpolation type", options=('linear', 'spline', 'polynomial'), index=0)
        
    with col2:
        ti_input = st.text_input(label='time values',
                                 help = 'space-separated, same amount as sensor values!',
                                 value=ti_std_str,
                                 placeholder="please input time values")
    with col3:
        yi_input = st.text_input(label='sensor values',
                                 help = 'space-separated, same amount as time values!',
                                 value=yi_std_str,
                                 placeholder="please input sensor values")
    
    ti = string_to_list(ti_input)
    tmin = min(ti)
    tmax = max(ti)
    with col4:
        t0 = st.slider(
                't\N{SUBSCRIPT ZERO}',
                min_value=float(tmin),
                max_value=float(tmax),
                value=(tmax+tmin)/2*2/3 # sets t0 to 2/3 of interval
            )
    
    if (t0 < tmin) | (t0 > tmax):
        st.warning('t\N{SUBSCRIPT ZERO} must be within given set of time values!')
    
    #col1,col2 = st.columns([1,3])
    if interptype == 'polynomial':
        deg = st.slider('Create polynomail of degree: ', min_value=0, max_value=20, value=(len(ti)-1))
    else:
        deg = 0
    if debug:
        st.write(deg)
    # update the data
    t_interp, y_interp, ft0, ti, yi, factors = update_data(interptype, t0, ti_input, yi_input, res, deg)
    
    if debug:
        st.write(factors)
    description = make_description(interptype,factors)
    [col1, col2] = st.columns(2)
    # with col1:
    #     st.markdown(description)
    # with col2:
    #     st.markdown(r"$$f(t_0) \approx " + str(round(ft0, 3)) + r"$$")
    st.markdown(r"Approximated $$f(t_0) \approx " + str(round(ft0, 3)) + r"$$. " + description)
    
    if interptype == 'polynomial':
        with st.expander('Show equation solved by np.polyfit using a Vandermonde matrix'):
            textsize = st.sidebar.selectbox(
                label="font size of formula",
                options=[r'\normalsize', r'\small', r'\footnotesize', r'\scriptsize', r'\tiny'],
                index=2)
            if deg == len(ti)-1:
                st.markdown(print_equations(ti, yi, factors, textsize))
            else:
                st.markdown("Vandermonde matrix is only exact for polynomial of degree n-1, in this case "
                            + str(len(ti)-1) + ".")
    
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
        update_plot(ti, yi, t0, ft0, t_interp, y_interp, toggle_interp,ti_input,yi_input,ticks_on)
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
