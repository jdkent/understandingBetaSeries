''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``hrf`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
import numpy as np
from nistats import hemodynamic_models
from statsmodels.tsa.arima_process import ArmaProcess
import pandas as pd
from scipy.optimize import minimize

from bokeh.layouts import grid
from bokeh.core.properties import value
from bokeh.palettes import Dark2
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Label, Div, Toggle, LabelSet
from bokeh.models.widgets import Slider
from bokeh.plotting import figure
from bokeh.models.glyphs import Segment


def generate_2voxels_signal(tr=1, corr=0, n_trials=15,
                            initial_guess=None,
                            design_resolution=0.1,
                            onset_offset=0,
                            timepoints=200):
    np.random.seed(12345)
    corr_mat = np.array([[1, corr], [corr, 1]])

    trial_interval = int(timepoints / n_trials)
    onsets = np.array(range(onset_offset, timepoints-10, trial_interval))

    # want the mean betas of each voxel to be one
    gnd_means = np.ones(2)

    # continue while loop while the target correlations
    # are more than 0.1 off
    c_wrong = True
    c_tol = 0.1

    while c_wrong:
        # generate betas
        if initial_guess is None:
            initial_guess = np.random.multivariate_normal(
                gnd_means,
                corr_mat,
                size=(onsets.shape[0]),
                tol=0.00005
            )

        sim_betas = minimize(
            _check_data,
            initial_guess,
            args=(corr_mat,),
            method='BFGS',
            tol=1e-10
        ).x

        # reshape the output (comes out 1-dimensional)
        sim_betas = sim_betas.reshape(initial_guess.shape)

        corr_error = _check_data(
            sim_betas,
            corr_mat,
        )

        c_wrong = c_tol < corr_error
        initial_guess = None

    mean_fix = 1 - sim_betas.mean(axis=0)
    # ensure each beta series has average of 1.
    betas = sim_betas + mean_fix

    onsets_res = (onsets // design_resolution).astype(int)
    duration_res = int(timepoints / design_resolution)
    stim_duration_res = int(0.5 / design_resolution)
    sampling_rate = int(tr / design_resolution)

    X = np.zeros((duration_res, onsets.shape[0]))

    for idx, onset in enumerate(onsets_res):
        # set the design matrix
        X[onset:onset+stim_duration_res, idx] = 1
        X[:, idx] = np.convolve(
            X[:, idx], hemodynamic_models._gamma_difference_hrf(
                tr, oversampling=sampling_rate))[0:X.shape[0]]

    # downsample X so it's back to TR resolution
    X = X[::sampling_rate, :]

    region1 = np.squeeze(X @ betas[:, 0])
    region2 = np.squeeze(X @ betas[:, 1])

    return onsets, betas, region1, region2


def generate_noise(timepoints=200, scale=0.01):
    np.random.seed(12345)
    # make the noise component
    rho = 0.12
    ar = np.array([1, -rho])  # statmodels says to invert rho
    ap = ArmaProcess(ar)
    err = ap.generate_sample(timepoints, scale=scale, axis=0)

    return err


def add_components(conditions, noise=None):
    if noise:
        conditions.append(noise)
    Y = np.sum(conditions, axis=0)

    return Y


def _check_data(x, target_corr_mat):
    corr_mat_obs = np.corrcoef(x.T)
    corr_error = _check_corr(corr_mat_obs, target_corr_mat)

    return corr_error


def _check_corr(corr_mat_obs, corr_mat_gnd):
    return np.sum(np.abs(corr_mat_obs - corr_mat_gnd)) / 2


# Set up data
corr_a = 0.0
corr_b = 0.8
timepoints = 200


# values for condition A
onsets_a, betas_a, region_1a, region_2a = generate_2voxels_signal(
    corr=corr_a, timepoints=timepoints)
# values for condition B
onsets_b, betas_b, region_1b, region_2b = generate_2voxels_signal(
    corr=corr_b, timepoints=timepoints, onset_offset=4)
# full signal
Y_region_1 = add_components([region_1a, region_1b])
Y_region_2 = add_components([region_2a, region_2b])

df_r1 = pd.DataFrame.from_dict({"Y": Y_region_1,
                                "A": region_1a,
                                "B": region_1b})

df_r2 = pd.DataFrame.from_dict({"Y": Y_region_2,
                                "A": region_2a,
                                "B": region_2b})
source_r1 = ColumnDataSource(df_r1)
source_r2 = ColumnDataSource(df_r2)
# label_source
betas_a_rnd = np.round(betas_a, 2)
betas_b_rnd = np.round(betas_b, 2)
df_ls_a = pd.DataFrame.from_dict({"onsets": onsets_a,
                                  "betas_r1": betas_a_rnd[:, 0],
                                  "betas_r2": betas_a_rnd[:, 1]})
df_ls_b = pd.DataFrame.from_dict({"onsets": onsets_b,
                                  "betas_r1": betas_b_rnd[:, 0],
                                  "betas_r2": betas_b_rnd[:, 1]})
source_la = ColumnDataSource(df_ls_a)
source_lb = ColumnDataSource(df_ls_b)


# Set up plots for time series
plot_r1 = figure(plot_height=200, plot_width=800, title="Region 1",
                 tools="crosshair,reset,save,wheel_zoom",
                 x_range=[0, timepoints], y_range=[-.2, 0.8])

plot_r2 = figure(plot_height=200, plot_width=800, title="Region 2",
                 tools="crosshair,reset,save,wheel_zoom",
                 x_range=[0, timepoints], y_range=[-0.2, 0.8])

plot_r1.xaxis.visible = plot_r2.xaxis.visible = False
plot_r1.xaxis.axis_label = plot_r2.xaxis.axis_label = "Volumes"

for plot, source in zip([plot_r1, plot_r2], [source_r1, source_r2]):
    for name, color in zip(df_r1.columns, Dark2[3]):
        if "Y" in name:
            line_width = 4
            line_alpha = 0.6
        else:
            line_width = 2
            line_alpha = 1.

        plot.line(x='index',
                  y=name,
                  line_width=line_width,
                  line_alpha=line_alpha,
                  source=source,
                  color=color,
                  legend=value(name))

    plot.legend.location = "top_left"
    plot.legend.orientation = "horizontal"
    plot.legend.click_policy = "hide"

trial_type_a_corr = Label(x=159, y=175, x_units='screen', y_units='screen',
                          text='Trial Type A Correlation: {corr}'.format(corr=corr_a),
                          render_mode='css', border_line_color='black', border_line_alpha=1.0,
                          background_fill_color='white', background_fill_alpha=1.0)

trial_type_b_corr = Label(x=385, y=175, x_units='screen', y_units='screen',
                          text='Trial Type B Correlation: {corr}'.format(corr=corr_b),
                          render_mode='css', border_line_color='black', border_line_alpha=1.0,
                          background_fill_color='white', background_fill_alpha=1.0)

plot_r2.add_layout(trial_type_a_corr)
plot_r2.add_layout(trial_type_b_corr)

# Set up widgets
data_title = Div(text="<b>Correlation Settings</b>",
                 style={'font-size': '100%'}, width=200, height=30)
corr_a_widget = Slider(title="Trial Type A Correlation", value=corr_a, start=-1, end=1, step=0.1)
corr_b_widget = Slider(title="Trial Type B Correlation", value=corr_b, start=-1, end=1, step=0.1)
noise_widget = Slider(title="noise", value=0, start=0, end=0.1, step=0.01)


widgets = [data_title,
           corr_a_widget,
           corr_b_widget,
           noise_widget]

beta_a_toggle = Toggle(label="Show A Betas", button_type="success", active=True)
beta_b_toggle = Toggle(label="Show B Betas", button_type="success", active=True)


for y0, y1, plot, beta_key, in zip([-0.2, 0.04],
                                   [0, 0.08],
                                   [plot_r1, plot_r2],
                                   ['betas_r1', 'betas_r2']):
    for idx, (onsets, betas, toggle, source) in enumerate(zip([onsets_a, onsets_b],
                                                              [betas_a, betas_b],
                                                              [beta_a_toggle, beta_b_toggle],
                                                              [source_la, source_lb])):
        y0s = y0 * len(onsets)
        y1s = y1 * len(onsets)
        lbls = LabelSet(x='onsets', y=0.35,
                        text=beta_key, source=source,
                        text_color=Dark2[3][idx+1])
        plot.add_layout(lbls)
        toggle.js_link('active', lbls, 'visible')
        plot.segment(x0=onsets, x1=onsets, y0=y0s, y1=y1s,
                     color=Dark2[3][idx+1], line_width=4)

# set up plots for correlations
plot_a = figure(plot_height=400, plot_width=400, title="Beta Series Correlation: A",
                tools="crosshair,reset,hover,save,wheel_zoom")
plot_b = figure(plot_height=400, plot_width=400, title="Beta Series Correlation: B",
                tools="crosshair,reset,hover,save,wheel_zoom")

plot_a.scatter(x='betas_r1', y='betas_r2', fill_color=Dark2[3][1],
               source=source_la, line_color=None, size=15)
plot_b.scatter(x='betas_r1', y='betas_r2', fill_color=Dark2[3][2],
               source=source_lb, line_color=None, size=15)


def update(attrname, old, new):
    # values for condition A
    _, betas_a, region_1a, region_2a = generate_2voxels_signal(
        corr=corr_a_widget.value, timepoints=timepoints)
    # values for condition B
    _, betas_b, region_1b, region_2b = generate_2voxels_signal(
        corr=corr_b_widget.value, timepoints=timepoints, onset_offset=4)
    # full signal
    region_1_list = [region_1a, region_1b]

    region_2_list = [region_2a, region_2b]

    if noise_widget.value > 0:
        region_1_noise = generate_noise(timepoints=timepoints, scale=noise_widget.value)
        region_2_noise = generate_noise(timepoints=timepoints, scale=noise_widget.value)
        region_1_list.append(region_1_noise)
        region_2_list.append(region_2_noise)

    Y_region_1 = add_components(region_1_list)
    Y_region_2 = add_components(region_2_list)

    betas_a_rnd = np.round(betas_a, 2)
    betas_b_rnd = np.round(betas_b, 2)
    s = slice(0, len(source_r1.data['Y']))
    source_r1.patch({"Y": [(s, Y_region_1)],
                     "A": [(s, region_1a)],
                     "B": [(s, region_1b)]
                     })
    source_r2.patch({"Y": [(s, Y_region_2)],
                     "A": [(s, region_2a)],
                     "B": [(s, region_2b)]
                     })
    s2 = slice(0, len(source_la.data['onsets']))
    source_la.patch({"betas_r1": [(s2, betas_a_rnd[:, 0])],
                          "betas_r2": [(s2, betas_a_rnd[:, 1])]
                          })
    s3 = slice(0, len(source_lb.data['onsets']))
    source_lb.patch({"betas_r1": [(s3, betas_b_rnd[:, 0])],
                          "betas_r2": [(s3, betas_b_rnd[:, 1])]
                          })

    trial_type_a_corr.text = 'Trial Type A Correlation: {corr}'.format(corr=corr_a_widget.value)
    trial_type_b_corr.text = 'Trial Type B Correlation: {corr}'.format(corr=corr_b_widget.value)
# def update_data(attrname, old, new):
#     # Generate the new curve
#     Y = generate_signal(go_onset=go_onset.value,
#                         ss_onset=ss_onset.value,
#                         fs_onset=fs_onset.value,
#                         go_pwr=go_pwr.value,
#                         ss_pwr=ss_pwr.value,
#                         fs_pwr=fs_pwr.value,
#                         noise=noise.value)

#     s = slice(0, len(source.data['Y']))
#     source.patch({"Y": [(s, Y)]})
#     tot_err = np.sum(np.abs(source.data['Y'] - source.data['Y_est']))
#     tracker.text = 'Total Error: {err}'.format(err=tot_err)
#     go_marker.x = go_onset.value
#     ss_marker.x = ss_onset.value
#     fs_marker.x = fs_onset.value


# def update_est(attrname, old, new):
#     go_est = generate_signal(go_onset=go_beta_onset.value,
#                              go_pwr=go_beta.value,
#                              ss_pwr=0, fs_pwr=0)
#     ss_est = generate_signal(ss_onset=ss_beta_onset.value,
#                              ss_pwr=ss_beta.value,
#                              go_pwr=0, fs_pwr=0)
#     fs_est = generate_signal(fs_onset=fs_beta_onset.value,
#                              fs_pwr=fs_beta.value,
#                              go_pwr=0, ss_pwr=0)
#     Y_est = go_est + ss_est + fs_est

#     s = slice(0, len(source.data['Y']))

#     source.patch({"go_estimate": [(s, go_est)],
#                   "successful_stop_estimate": [(s, ss_est)],
#                   "failed_stop_estimate": [(s, fs_est)],
#                   "Y_est": [(s, Y_est)]})
#     tot_err = np.sum(np.abs(source.data['Y'] - source.data['Y_est']))
#     tracker.text = 'Total Error: {err}'.format(err=tot_err)


for w in widgets[1:]:
    w.on_change('value', update)

# for w in est_widgets[1:]:
#     w.on_change('value', update_est)


# Set up layouts and add to document

data_inputs = column(widgets)

curdoc().add_root(grid([[plot_r1], [plot_r2, [beta_a_toggle, beta_b_toggle]], [plot_a, plot_b], [data_inputs]]))
curdoc().title = "My Awesome Task"
