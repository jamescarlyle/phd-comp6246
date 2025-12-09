import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")

with app.setup:
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as ptch
    import scipy.signal as si
    import scipy.stats as st
    import scipy.optimize as op
    import sklearn.metrics as sm
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    import sklearn.preprocessing as pp
    import tensorflow as tf
    from tensorflow.keras import models, optimizers
    from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D, LSTM, Dropout, Dense, Bidirectional, Flatten, GlobalAveragePooling1D, Reshape
    from functools import partial

    CSV_COLS = ['timestamp', 'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'label']
    ACTIVITIES = ['walking', 'running', 'shuffling', 'standing', 'sitting', 'lying', 'stairs']  
    UNWANTED_LABELS = [10, 13, 14, 130, 140]
    BACK_AXES = ['back_x', 'back_y', 'back_z']
    THIGH_AXES = ['thigh_x', 'thigh_y', 'thigh_z']
    AXES = np.concatenate((BACK_AXES, THIGH_AXES))
    BACK_X_FEATURES = ['back_x_10p', 'back_x_median', 'back_x_90p', 'back_x_std']
    BACK_Y_FEATURES = ['back_y_10p', 'back_y_median', 'back_y_90p', 'back_y_std']
    BACK_Z_FEATURES = ['back_z_10p', 'back_z_median', 'back_z_90p', 'back_z_std']
    BACK_ENMO_MS = ['back_enmo_median', 'back_enmo_std']
    BACK_ENMO_P = ['back_enmo_10p', 'back_enmo_90p']
    BACK_ENMO_FEATURES = BACK_ENMO_MS + BACK_ENMO_P
    BACK_FEATURES = BACK_X_FEATURES + BACK_Y_FEATURES + BACK_Z_FEATURES + BACK_ENMO_FEATURES
    BACK_10_FEATURES = BACK_X_FEATURES + BACK_Z_FEATURES + BACK_ENMO_MS
    THIGH_X_FEATURES = ['thigh_x_10p', 'thigh_x_median', 'thigh_x_90p', 'thigh_x_std']
    THIGH_Y_FEATURES = ['thigh_y_10p', 'thigh_y_median', 'thigh_y_90p', 'thigh_y_std']
    THIGH_Z_FEATURES = ['thigh_z_10p', 'thigh_z_median', 'thigh_z_90p', 'thigh_z_std']
    THIGH_ENMO_MS = ['thigh_enmo_median', 'thigh_enmo_std']
    THIGH_ENMO_P = ['thigh_enmo_10p', 'thigh_enmo_90p']
    THIGH_ENMO_FEATURES = THIGH_ENMO_MS + THIGH_ENMO_P
    THIGH_FEATURES = THIGH_X_FEATURES + THIGH_X_FEATURES + THIGH_Z_FEATURES + THIGH_ENMO_FEATURES
    THIGH_10_FEATURES = THIGH_X_FEATURES + THIGH_Z_FEATURES + THIGH_ENMO_MS
    ALL_FEATURES = BACK_FEATURES + THIGH_FEATURES
    SENSOR_THRESHOLD = 8
    SAMPLE_RATE = 50
    WINDOW_SIZE = 2
    WINDOW_OVERLAP = 1
    INPUT_SHAPE = (SAMPLE_RATE * WINDOW_SIZE, len(AXES))
    EPOCHS = 10
    pd.set_option('display.max_rows', 100)
    label_encoder = pp.LabelEncoder()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Pre-processing and exploration
    ## Function definitions
    """)
    return


@app.function
def load_data(filepath):
        """Load data from csv files in a filepath to a dictionary of dataframes.

        Args:
            filepath: The name of the filepath to load data from.

        Returns:
            A new dictionary of dataframes, keyed by the filename without the extension.
        """
        _dataframes = []
        for filename in os.listdir(filepath):
            if filename.endswith('.csv'):
                _dataframe = pd.read_csv(filepath+'/'+filename, usecols=CSV_COLS, parse_dates=[0])
                _dataframe['sensor'] = os.path.splitext(os.path.basename(filename))[0]
                _dataframes.append(_dataframe)
        return pd.concat(objs=_dataframes, axis=0)


@app.cell
def _():
    train_data = load_data('./MLT-CW-Dataset')
    test_data = load_data('./MLT-CW-Dataset/test-set')
    return test_data, train_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Report the class balance of the whole dataset.
    """)
    return


@app.cell
def _(test_data, train_data):
    print(f'Training dataframe shape: {train_data.shape}')
    print(f'Testing dataframe shape: {test_data.shape}')
    print(f'Columns: {train_data.columns}')
    return


@app.cell
def _(train_data):
    # Report the class balance of the whole dataset.
    _value_counts = train_data['label'].value_counts().sort_values(ascending=False)
    _value_counts.plot(kind='bar', figsize=(8, 5))
    plt.title('Class Distribution in Training Data')
    plt.xlabel('Activities')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Write a function to drop data labelled with any type of cycling.
    """)
    return


@app.function
def drop_labelled_rows(dataframe, labels):
    """Drop rows from a dataframe based on row label inplace, i.e. the dataframe is modified.

    Args:
        df: The dataframe.
        labels: An array of string label names.

    Returns:
        A new dictionary of dataframes, keyed by the filename without the extension.
    """
    mask = dataframe['label'].isin(labels)
    return dataframe[~mask]


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Apply it to the dataset and report the new dataset size.
    """)
    return


@app.cell
def _(test_data, train_data):
    unwanted_cycling = drop_labelled_rows(train_data, UNWANTED_LABELS)
    test_unwanted_cycling = drop_labelled_rows(test_data, UNWANTED_LABELS)
    print(f'Training dataframe shape: {unwanted_cycling.shape}')
    print(f'Testing dataframe shape: {test_unwanted_cycling.shape}')
    del train_data
    del test_data
    return test_unwanted_cycling, unwanted_cycling


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Write a function to merge stairs in a single class with code 9
    """)
    return


@app.function
def replace_values(dataframe, column, existing_values, replacement_value):
    """Replace values of a column in a dataframe.

    Args:
        dataframe: The dataframe to be modified.
        column: A string name of the column to modify values in.
        existing_values: An array of strings to be replaced.
        replacement_value: A string to substitute.

    Returns:
        A new dataframe with the column values replaced.
    """
    new_df = dataframe.copy()
    # Replace the existing values in the specified column with the replacement_value.
    new_df[column] = new_df[column].replace(existing_values, replacement_value)
    return new_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Apply it to the dataset
    """)
    return


@app.cell
def _(test_unwanted_cycling, unwanted_cycling):
    merged_stairs = replace_values(unwanted_cycling, 'label', [4,5], 9)
    test_merged_stairs = replace_values(test_unwanted_cycling, 'label', [4,5], 9)
    print(f'Training dataframe shape after merging stairs: {unwanted_cycling.shape}')
    print(f'Testing dataframe shape after merging stairs: {test_unwanted_cycling.shape}')
    del unwanted_cycling
    del test_unwanted_cycling
    return merged_stairs, test_merged_stairs


@app.cell
def _(merged_stairs):
    _value_counts = merged_stairs['label'].value_counts().sort_values(ascending=False)
    _value_counts.plot(kind='bar', figsize=(8, 5))
    plt.title('Class Distribution after cycling removed and stairs merged')
    plt.xlabel('Activities')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Use appropriate sampling and visualisations for Walking.
    Start by selecting a sensor set and a sample of rows, limited to a meaningful number, say 8 seconds:
    """)
    return


@app.cell
def _(merged_stairs):
    _walking_viz = merged_stairs[(merged_stairs['label'] == 1) & (merged_stairs['sensor'] == 'S012')].iloc[::1].head(400)
    _fig, _ax = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    for _i, _col in enumerate(AXES):
        _ax[_i].plot(_walking_viz['timestamp'], _walking_viz[_col], linewidth=0.5, label="Raw Data")
        # Use built-in Savitzky-Golay filter for preserving shape while denoising the data.
        _ax[_i].plot(_walking_viz['timestamp'], si.savgol_filter(_walking_viz[_col], window_length=5, polyorder=1), linewidth=0.8, label="Savitzky-Golay Filter")
        _ax[_i].set_title(_col)
        _ax[_i].set_ylabel('Acceleration')

    _ax[-1].legend(loc="upper right")
    _ax[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.show()
    del _walking_viz
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Data Cleaning and Preparation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Analyse and report the data quality of the S007 subset
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### There is an issue with all sensors - some accelerometer data is constant across timestamps
    This is particularly true for S007, notably because all axes of both sensors show a constant.
    Identify by looking for discontinuities of values, grouping by sensor and then axis. Once groups of values are identified, aggregate each group to capture the first and last timestamps of the series, and the number of rows in between. There are around 250 instances where a sensor is reading a perfectly constant acceleration of more than 1 second for a sampling frequency of 50Hz. It would be impossible for a human to achieve this, and must be caused by the sensor getting 'stuck' in a particular position.
    """)
    return


@app.cell
def _(merged_stairs):
    _results = {}
    # First group by sensor, then by axis.
    for sensor_name in merged_stairs['sensor'].unique():
        _sensor_df = merged_stairs[merged_stairs['sensor'] == sensor_name].copy().reset_index(drop=True)
        if len(_sensor_df) < 101:
            continue

        _sensor_results = {}
        for _col in AXES:
            _groups_s = (_sensor_df[_col] != _sensor_df[_col].shift()).cumsum()
            _grouped_s = _sensor_df.groupby([_groups_s, _col]).agg(
                _start_time=('timestamp', 'first'),
                _end_time=('timestamp', 'last'),
                _count=('timestamp', 'size')
            ).reset_index(drop=True)
            _filtered_s = _grouped_s[_grouped_s['_count'] > 100]
            _sensor_results[_col] = _filtered_s[['_start_time', '_end_time', '_count']]
        _results[sensor_name] = _sensor_results

    # Flatten to single DataFrame with sensor/axis as first columns.
    _flattened_rows = []
    for sensor, axes_data in _results.items():
        for axis, df in axes_data.items():
            if not df.empty:
                # Add sensor and axis columns to each row
                df_with_meta = df.copy()
                df_with_meta.insert(0, 'axis', axis)
                df_with_meta.insert(0, 'sensor', sensor)
                _flattened_rows.append(df_with_meta)

    _flattened_df = pd.concat(_flattened_rows, ignore_index=True).sort_values(ascending=False,by='_count')
    print (_flattened_df._count.sum())
    print(_flattened_df.head(10).to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Information loss with ENMO
    There are many data rows where the Euclidean Norm Minus One (ENMO) is less than zero. This could be possible during periods of downward step, where the x-axis (vertical) measured acceleration is less than 1 (i.e. the thigh is being forced down) while y- and z-axis acceleration is close to zero, or could be due to incorrect sensor calibration. Truncating negative values to zero (standard in much research) causes information loss and reduces the ability to identify activities, so for label fitting, plain ENMO is used, not max(ENMO, 0).
    """)
    return


@app.cell
def _(merged_stairs):
    _enmo_mask = (np.linalg.norm(merged_stairs[BACK_AXES], axis=1) - 1) < 0
    print(f'Percentage of back sensor readings with ENMO < 0 (normally truncated): {np.count_nonzero(_enmo_mask) * 100/len(merged_stairs):.2f}%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### There are also discontinuities in the data
    This means that windows can either be a fixed number of rows or length of time, not both. For example, sensor S006 shows a 3-second gap at 00h03m08s:
    """)
    return


@app.cell
def _(merged_stairs):
    _mask = (
        (merged_stairs.sensor == 'S006') &
        (merged_stairs.timestamp >= pd.Timestamp('2019-01-12 00:03:08.360')) &
        (merged_stairs.timestamp <= pd.Timestamp('2019-01-12 00:03:11.610'))
    )
    print(merged_stairs.loc[_mask])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Remove the particularly egregious S007 constant data
    """)
    return


@app.cell
def _(merged_stairs):
    _mask = ~(
        (merged_stairs.sensor == 'S007') &
        (merged_stairs.timestamp >= pd.Timestamp('2019-01-17 00:00:41.840')) &
        (merged_stairs.timestamp <= pd.Timestamp('2019-01-17 00:01:25.660'))
    )
    merged_without_s007_flat = merged_stairs.loc[_mask]
    return (merged_without_s007_flat,)


@app.cell
def _(merged_without_s007_flat):
    s007_data = merged_without_s007_flat[merged_without_s007_flat.sensor == 'S007']
    non_s007_data = merged_without_s007_flat[merged_without_s007_flat.sensor != 'S007']
    return non_s007_data, s007_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Spurious outlier data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    S007 accelerometer data show unreasonable outliers seen in these violin plots; the AX3 sensor is rated to +-8g. However, the means and distribution of the data without outliers is similar, suggesting that only a few timestamps are erroneous.
    """)
    return


@app.cell
def _(non_s007_data, s007_data):
    _fig, _ax = plt.subplots(ncols=1, nrows=6, figsize=(10, 6), sharex=True)
    _fig.suptitle('Violin plots of axes for s007 vs non_s007')
    for _i, _col in enumerate(AXES):
        _ax[_i].violinplot(non_s007_data[_col].values.T, vert=False)
        _ax[_i].violinplot(s007_data[_col].values.T, vert=False)
        _ax[_i].set_xlabel(_col)
    non_patch = ptch.Patch(color='tab:blue',  alpha=0.6, label='non-S007')
    s007_patch = ptch.Patch(color='tab:orange', alpha=0.6, label='S007')
    _fig.suptitle('Violin plots of axes for s007 vs non_s007')
    _ax[0].legend(handles=[non_patch, s007_patch], loc="upper right")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(non_s007_data, s007_data):
    # Drop unneeded dataframes to free memory.
    del s007_data
    del non_s007_data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Remove spurious outliers
    Outliers are removed by setting any sensor acceleration values where the value is above the rated maximum (8g) for the AX3 accelerometer, in any axis, to null, and then interpolating any nulls linearly and with forward- and backward-fill.
    """)
    return


@app.function
def interpolate_outliers(df, columns, threshold, method):
    outliers_removed = df.copy()
    outliers_removed[columns] = np.where(
        outliers_removed[columns].abs() <= threshold,
        outliers_removed[columns], np.nan
    )
    outliers_removed[columns] = outliers_removed[columns].interpolate(method=method).ffill().bfill()
    return outliers_removed


@app.cell
def _(merged_stairs, merged_without_s007_flat, test_merged_stairs):
    cleaned = interpolate_outliers(merged_without_s007_flat, AXES, SENSOR_THRESHOLD, 'linear')
    test_cleaned = interpolate_outliers(test_merged_stairs, AXES, SENSOR_THRESHOLD, 'linear')
    cleaned_unique_labels = sorted(cleaned.label.unique())
    label_encoder.fit(cleaned_unique_labels)
    del merged_without_s007_flat
    del merged_stairs
    del test_merged_stairs
    return cleaned, cleaned_unique_labels, test_cleaned


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Show distribution of acceleration with outliers removed
    """)
    return


@app.cell
def _(cleaned):
    _fig, _ax = plt.subplots(ncols=1, nrows=6, figsize=(10, 6), sharex=True)
    _fig.suptitle('Violin plots of axes for outliers removed')
    for _i, _col in enumerate(AXES):
        _ax[_i].violinplot(cleaned[_col].values.T, vert=False)
        _ax[_i].set_xlabel(_col)
    _no_outliers = ptch.Patch(color='tab:blue',  alpha=0.6, label='outliers-removed')
    _ax[0].legend(handles=[_no_outliers], loc="upper right")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(cleaned):
    # Box Plots of sensor data by X,Y,Z for back, thigh locations
    _fig, _axes = plt.subplots(2, 3, figsize=(10, 8), sharey=True, sharex=True)
    _fig.suptitle('Univariate analysis of acceleration by X,Y,Z for activity and back, thigh locations with outliers removed')
    _labels = sorted(cleaned['label'].unique())
    for _i, _ax in enumerate(_axes.flatten()):
        data = [cleaned[cleaned['label'] == _lbl][AXES[_i]].values for _lbl in _labels]
        _bp =_ax.boxplot(data, tick_labels=_labels, showfliers=False, patch_artist=True)
        _ax.set_title(AXES[_i])
        [_median.set_color('black') for _median in _bp['medians']]
        for _patch, _color in zip(_bp['boxes'], plt.cm.tab10.colors):
            _patch.set_facecolor(_color)
            _patch.set_edgecolor('black')
    _fig.supxlabel('Activity')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It's clear from the distribution in raw accelerometer readings from sensors that some of the activities lead to sensor readings with very distinctive characteristics. For example, class 2 (running) has much greater range in minimum and maximum values compared with class 1 (walking). On the other hand, the values for walking and class 9 (stairs) are very similar, suggesting that simple separation by min, max, mean and standard deviation between these two classes is going to be very difficult.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Report on dataset size
    """)
    return


@app.cell
def _(cleaned):
    print(cleaned.shape)
    return


@app.cell
def _(cleaned):
    print(cleaned.groupby('label').size())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ##Write a function that receives the dataset and outputs sliding windows of 2 seconds size
    """)
    return


@app.function
def generate_time_windows(df, window_length, overlap):
    """Generate windows (slices of a dictionary of dataframes). By iterating over a sensor grouping,
    no window can span two sensors.

    Args: 
        df: A complete dataframe containing multiple sensor data.
        window_length: Length of each window to be created, in seconds.
        overlap: Overlap of the next window over the previous, in seconds.

    Returns:
        An array of windows (dataframes), aggregated across the dictionary of dataframes passed in.
    """    
    window_size = pd.Timedelta(seconds=window_length)
    step_size = pd.Timedelta(seconds=(window_length - overlap))
    # windows is an array of dataframes.
    windows = []
    # Iterate by sensor, so that windows can't span sensors (wouldn't make sense).
    for sensor_name in df['sensor'].unique():
        sensor_df = df[df['sensor'] == sensor_name].copy().reset_index(drop=True)
        start_time = sensor_df.timestamp.min()
        end_time = sensor_df.timestamp.max()
        current_start = start_time
        while current_start + window_size <= end_time:
            current_end = current_start + window_size
            window_slice = sensor_df[(sensor_df.timestamp >= current_start) & (sensor_df.timestamp < current_end)]
            # Unfortunately the discontinuities / gaps in sensor data timestamps mean that some time-based slices are empty.
            if not window_slice.empty:
                windows.append(window_slice)
            current_start += step_size
    return windows


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Report the number of data points with this window size.
    """)
    return


@app.cell
def _(cleaned, test_cleaned):
    windows_array = generate_time_windows(df=cleaned, window_length=WINDOW_SIZE, overlap=WINDOW_OVERLAP)
    print(len(windows_array))
    test_windows_array = generate_time_windows(df=test_cleaned, window_length=WINDOW_SIZE, overlap=WINDOW_OVERLAP)
    print(len(test_windows_array))
    print(f'Train data window count: {len(windows_array)}')
    print(f'Test data window count: {len(test_windows_array)}')
    return test_windows_array, windows_array


@app.cell
def _(windows_array):
    windows_array[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Pipelines
    ## Establish a baseline with an initial set of 10 features
    A solid superset of features is built in one go. 10 of these are to be selected for baseline evaluation. Looking at boxplot univariate analysis, there are clear differences in max and min values by activity, but some of the data are spurious and truncated at +-8g, so max and min is less effective (and also susceptible to participant body dimensions). Instead, the 10-percentile and 90-percentile is taken, as well as the median and standard deviation.
    """)
    return


@app.function
def stats_from_frame(df, norm_name, column_names):
    """Calculate stats from a np.array.

    Args: 
        data_frame: The dataframe containing the columns to be aggregated.
        norm_name: A string to be used for the norm column names.
        column_names: An array of strings containing column names to be used.

    Returns:
        An array of scalars for 10%, 50% (median), 90%tile, standard deviation for each column, 
        and the same for the Euclidian norm.
        An array of column names for the array of scalars.
    """
    norm_array = np.linalg.norm(df[column_names], axis=1)
    stats = []
    titles = []
    for column in column_names:
        # stats.append(df[column_names].quantile([0.1, 0.5, 0.9]))
        stats.extend(np.percentile(a=df[column], q=[10, 50, 90]))
        # Because the deviation is calculated from the entire data set, not a sample, ddof (degrees of freedom) is 0.
        # If ddof=1, then std() returns Nan if only one row processed. Nan value is not allowed in kmeans analysis.
        stats.append(df[column].std(ddof=0))
        titles.extend([f'{column}_10p', f'{column}_median', f'{column}_90p', f'{column}_std'])
    stats.extend(np.percentile(a=norm_array, q=[10, 50, 90], axis=0))
    stats.append(norm_array.std(ddof=0))
    titles.extend([f'{norm_name}_10p', f'{norm_name}_median', f'{norm_name}_90p', f'{norm_name}_std'])
    return stats, titles


@app.function
def generate_window_summaries(windows, norm_name_1, axes_1, norm_name_2, axes_2):
    """Generate a dataframe containing a summary and statistics for each window.

    Args: 
        windows: An array of windows, where each window is a dataframe.
        back_features: An array of arrays containing column names to be used as features, normally data from a set of 3 axes.
        back_enmo: An array of arrays containing column names to be used as features.
        thigh_features: An array of arrays containing column names to be used as features, normally data from a set of 3 axes.
        thigh_enmo: An array of arrays containing column names to be used as features.

    Returns:
        An array of scalars for mean, standard deviation, min, max for each column, 
        and the same for the Euclidian norm of the columns combined.
    """
    window_summaries = []
    for _window in windows:
        stats_1, titles_1 = stats_from_frame(_window, norm_name_1, axes_1)
        stats_2, titles_2 = stats_from_frame(_window, norm_name_2, axes_2)
        window_summaries.append(np.concatenate((
            # First timestamp of window.
            [_window.timestamp.min()], 
            # Separate back and thigh so that the function can generate a single set of norms generically. 
            stats_1,
            stats_2,
            # Choose the most popular label for the window.
            [_window.label.mode()[0]] 
        )))
    return pd.DataFrame(window_summaries, columns=['timestamp'] + titles_1 + titles_2 + ['label'])


@app.cell
def _(test_windows_array, windows_array):
    # stats_from_frame(windows_array[0], 'back_enmo', BACK_AXES)
    window_summaries = generate_window_summaries(windows_array, 'back_enmo', BACK_AXES, 'thigh_enmo', THIGH_AXES)
    test_window_summaries = generate_window_summaries(test_windows_array, 'back_enmo', BACK_AXES, 'thigh_enmo', THIGH_AXES)
    print(f'Training data window summaries: {window_summaries.shape}')
    print(f'Test data window summaries: {test_window_summaries.shape}')
    del windows_array
    del test_windows_array
    return test_window_summaries, window_summaries


@app.cell
def _(window_summaries):
    window_summaries.iloc[::10]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KMeans
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Baseline
    Begin by plotting the inertia (sum of squares of distance from points to corresponding cluster centres) against number of clusters to identify the right number of clusters.
    """)
    return


@app.cell
def _(window_summaries):
    _data = window_summaries[THIGH_10_FEATURES]
    K = range(1, 11)
    inertias = []

    for k in K:
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=100,
            random_state=42,
        )
        kmeans.fit(_data)
        inertias.append(kmeans.inertia_)   
    plt.figure()
    plt.plot(K, inertias, 'o-')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow method for KMeans")
    plt.xticks(K)
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The elbow method clearly shows an inflection point at 3 clusters. In an unsupervised setting, this would be the optimum cluster level, but in this case, an unsupervised method is being applied to labelled data with 7 known labels and so the K-means model is configured for 7 clusters.
    """)
    return


@app.cell
def _(window_summaries):
    # Create a k-means model with 7 clusters (since we are targetting assignment to 7 labels).
    kmeans_model = KMeans(n_clusters=7, init='k-means++', n_init='auto', random_state=42) #, algorithm='elkan'
    kmeans_model.fit(window_summaries[THIGH_10_FEATURES])
    # Create a copy of the windows data for k-means processing.
    _kmeans_data = window_summaries.copy()
    # Assign a cluster index to each window. Note that the kmeans labels are not the same as the groundtruth labels - 
    # they are an arbritary tag value.
    _kmeans_data['cluster'] = kmeans_model.labels_
    _unique_labels = sorted(_kmeans_data.label.unique())
    # The original confusion matrix.
    _confusion_raw = sm.confusion_matrix(label_encoder.transform(_kmeans_data.label), _kmeans_data.cluster)

    # Maximize diagonal sum, so that we can map from cluster tags to groundtruth labels. 
    # Ideally the diagonals (True Positives) have high values, with all of the other cells being low or zero.
    _row_indices, _col_indices = op.linear_sum_assignment(-1 * _confusion_raw)  
    # Map original cluster labels to new labels for best alignment.
    _mapping = {old: new for old, new in zip(_col_indices, _row_indices)}

    # Apply the remapping to cluster labels.
    _kmeans_data['cluster_mapped'] = label_encoder.inverse_transform(_kmeans_data['cluster'].map(_mapping))

    _precision_score = sm.precision_score(_kmeans_data['label'], _kmeans_data['cluster_mapped'], average=None)
    _recall_score = sm.recall_score(_kmeans_data['label'], _kmeans_data['cluster_mapped'], average=None)
    for _i, _label in enumerate(_unique_labels):
        print(f'For label {_label}: Precision {_precision_score[_i]:.2f}, Recall {_recall_score[_i]:.2f}')

    # Compute the confusion matrix with remapped clusters.
    _confusion_mapped = sm.confusion_matrix(_kmeans_data['label'], _kmeans_data['cluster_mapped'])
    _disp = sm.ConfusionMatrixDisplay(_confusion_mapped, display_labels=_unique_labels)
    _disp.plot(cmap=plt.cm.Blues)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    K-Means 7 clusters, BACK_ENMO_FEATURES, n_clusters=7, init='random', n_init=100, random_state=42.
    For label 1: Precision 0.82, Recall 0.34
    For label 2: Precision 1.00, Recall 0.53
    For label 3: Precision 0.09, Recall 0.15
    For label 6: Precision 0.00, Recall 0.00
    For label 7: Precision 0.69, Recall 0.93
    For label 8: Precision 0.27, Recall 0.22
    For label 9: Precision 0.11, Recall 0.19

    K-Means 7 clusters, BACK_ENMO_FEATURES, n_clusters=7, init='k-means++', n_init=100, random_state=42.
    For label 1: Precision 0.83, Recall 0.34
    For label 2: Precision 1.00, Recall 0.53
    For label 3: Precision 0.09, Recall 0.14
    For label 6: Precision 0.00, Recall 0.00
    For label 7: Precision 0.68, Recall 0.93
    For label 8: Precision 0.27, Recall 0.22
    For label 9: Precision 0.11, Recall 0.18

    K-Means 7 clusters, BACK_10_FEATURES, n_clusters=7, init='k-means++', n_init=100, random_state=42.
    For label 1: Precision 0.82, Recall 0.62
    For label 2: Precision 0.98, Recall 0.94
    For label 3: Precision 0.00, Recall 0.01
    For label 6: Precision 0.18, Recall 0.40
    For label 7: Precision 0.62, Recall 0.42
    For label 8: Precision 0.99, Recall 0.73
    For label 9: Precision 0.00, Recall 0.00

    K-Means 7 clusters, ALL_FEATURES, n_clusters=7, init='k-means++', n_init=100, random_state=42.
    For label 1: Precision 0.80, Recall 0.69
    For label 2: Precision 0.98, Recall 0.86
    For label 3: Precision 0.00, Recall 0.00
    For label 6: Precision 0.56, Recall 0.98
    For label 7: Precision 0.98, Recall 0.73
    For label 8: Precision 0.76, Recall 0.35
    For label 9: Precision 0.00, Recall 0.00

    K-Means 7 clusters, THIGH_10_FEATURES, n_clusters=7, init='k-means++', n_init=100, random_state=42.
    For label 1: Precision 0.75, Recall 0.45
    For label 2: Precision 0.99, Recall 0.75
    For label 3: Precision 0.00, Recall 0.00
    For label 6: Precision 0.62, Recall 0.96
    For label 7: Precision 0.95, Recall 0.98
    For label 8: Precision 1.00, Recall 0.55
    For label 9: Precision 0.06, Recall 0.17

    K-Means 7 clusters, BACK_10_FEATURES + THIGH_10_FEATURES, n_clusters=7, init='k-means++', n_init=100, random_state=42.
    For label 1: Precision 0.80, Recall 0.69
    For label 2: Precision 0.98, Recall 0.87
    For label 3: Precision 0.00, Recall 0.00
    For label 6: Precision 0.55, Recall 0.97
    For label 7: Precision 0.97, Recall 0.75
    For label 8: Precision 0.99, Recall 0.54
    For label 9: Precision 0.00, Recall 0.00

    K-Means 7 clusters, BACK_10_FEATURES + THIGH_10_FEATURES, n_clusters=7, init='k-means++', n_init=1, random_state=42.
    For label 1: Precision 0.80, Recall 0.38
    For label 2: Precision 0.99, Recall 0.81
    For label 3: Precision 0.00, Recall 0.00
    For label 6: Precision 0.62, Recall 0.96
    For label 7: Precision 0.97, Recall 0.75
    For label 8: Precision 0.84, Recall 0.60
    For label 9: Precision 0.20, Recall 0.74

    K-Means 7 clusters, BACK_10_FEATURES + THIGH_10_FEATURES, n_clusters=7, init='k-means++', n_init=1, random_state=42, algorithm='elkan'.
    For label 1: Precision 0.80, Recall 0.38
    For label 2: Precision 0.99, Recall 0.81
    For label 3: Precision 0.00, Recall 0.00
    For label 6: Precision 0.62, Recall 0.96
    For label 7: Precision 0.97, Recall 0.75
    For label 8: Precision 0.84, Recall 0.60
    For label 9: Precision 0.20, Recall 0.74
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Refinement
    """)
    return


@app.cell
def _():


    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random Forest
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Baseline
    """)
    return


@app.cell
def _(test_window_summaries, window_summaries):
    # Remove inappropriate columns from classification dataframe.
    _rf_x_train = window_summaries.drop(['timestamp', 'label'], axis=1, inplace=False)
    # Labels.
    _rf_y_train = window_summaries['label']  

    # Instantiate and train Random Forest classifier
    _rf = RandomForestClassifier(n_estimators=100, random_state=42,min_samples_leaf=1000)
    _rf.fit(_rf_x_train[ALL_FEATURES], _rf_y_train)

    # Predict labels for test set.
    _rf_pred_labels = _rf.predict(test_window_summaries[ALL_FEATURES])
    # Generate confusion matrix. 
    _confusion_rf = sm.confusion_matrix(test_window_summaries['label'], _rf_pred_labels)
    # Create a list of display labels.
    _display_labels = sorted(test_window_summaries['label'].unique())

    _precision_score = sm.precision_score(test_window_summaries['label'], _rf_pred_labels, average=None)
    _recall_score = sm.recall_score(test_window_summaries['label'], _rf_pred_labels, average=None)
    for _i, _label in enumerate(_display_labels):
        print(f'For label {_label}: Precision {_precision_score[_i]:.2f}, Recall {_recall_score[_i]:.2f}')

    # By default, the confusion matrix rows and columns will be sorted by label value, so we must provide the same sorting for display.
    _disp = sm.ConfusionMatrixDisplay(_confusion_rf, display_labels=_display_labels)
    _disp.plot(cmap=plt.cm.Blues)
    plt.show()

    del _rf_x_train
    del _rf_y_train
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Refinement
    Combinations tried to improve classes 3 and 9:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    n_estimators=100, ALL_FEATURES
    For label 1: Precision 0.93, Recall 0.84
    For label 2: Precision 0.99, Recall 0.96
    For label 3: Precision 0.33, Recall 0.56
    For label 6: Precision 0.89, Recall 0.86
    For label 7: Precision 1.00, Recall 1.00
    For label 8: Precision 1.00, Recall 1.00
    For label 9: Precision 0.44, Recall 0.58

    n_estimators=10, ALL_FEATURES
    For label 1: Precision 0.92, Recall 0.83
    For label 2: Precision 0.99, Recall 0.96
    For label 3: Precision 0.30, Recall 0.53
    For label 6: Precision 0.87, Recall 0.83
    For label 7: Precision 1.00, Recall 1.00
    For label 8: Precision 1.00, Recall 1.00
    For label 9: Precision 0.42, Recall 0.52

    n_estimators=100, THIGH_10_FEATURES
    For label 1: Precision 0.92, Recall 0.91
    For label 2: Precision 0.96, Recall 0.96
    For label 3: Precision 0.37, Recall 0.45
    For label 6: Precision 0.89, Recall 0.87
    For label 7: Precision 0.96, Recall 0.99
    For label 8: Precision 0.96, Recall 0.78
    For label 9: Precision 0.59, Recall 0.52

    n_estimators=100, THIGH_X_FEATURES
    For label 1: Precision 0.87, Recall 0.89
    For label 2: Precision 0.93, Recall 0.95
    For label 3: Precision 0.31, Recall 0.38
    For label 6: Precision 0.86, Recall 0.82
    For label 7: Precision 0.89, Recall 0.93
    For label 8: Precision 0.57, Recall 0.44
    For label 9: Precision 0.32, Recall 0.23

    n_estimators=100, criterion='log_loss', THIGH_X_FEATURES
    For label 1: Precision 0.87, Recall 0.89
    For label 2: Precision 0.93, Recall 0.95
    For label 3: Precision 0.32, Recall 0.38
    For label 6: Precision 0.86, Recall 0.82
    For label 7: Precision 0.89, Recall 0.94
    For label 8: Precision 0.57, Recall 0.41
    For label 9: Precision 0.32, Recall 0.23

    n_estimators=100, criterion='gini', max_depth=5, THIGH_X_FEATURES
    For label 1: Precision 0.86, Recall 0.95
    For label 2: Precision 0.94, Recall 0.96
    For label 3: Precision 0.37, Recall 0.43
    For label 6: Precision 0.89, Recall 0.82
    For label 7: Precision 0.86, Recall 0.99
    For label 8: Precision 0.72, Recall 0.15
    For label 9: Precision 0.00, Recall 0.00

    n_estimators=100, THIGH_X_FEATURES, min_samples_leaf=1000
    For label 1: Precision 0.86, Recall 0.94
    For label 2: Precision 0.90, Recall 0.96
    For label 3: Precision 0.35, Recall 0.37
    For label 6: Precision 0.87, Recall 0.84
    For label 7: Precision 0.85, Recall 0.99
    For label 8: Precision 0.87, Recall 0.08
    For label 9: Precision 0.00, Recall 0.00
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Convolutional Neural Network
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Neural networks need windows of constant size
    """)
    return


@app.function
def generate_fixed_windows(df, sample_rate, window_size, overlap):
    """Generate fixed size windows (slices of a dataframes).

    Args: 
        dataframes: A dictionary of dataframes, keyed by string (e.g. sensor name).
        window_length: Length of each window to be created, in seconds.
        overlap: Overlap of the next window over the previous, in seconds.

    Returns:
        An array of windows (dataframes), aggregated across the dictionary of dataframes passed in.
    """    
    window_rows = int(window_size * sample_rate)
    window_advance = int((window_size - overlap) * sample_rate)
    windows = []
    # Group by sensor, so that no window can span sensors.
    for sensor_name in df['sensor'].unique():
        sensor_df = df[df['sensor'] == sensor_name].copy().reset_index(drop=True)
        start_index = 0
        while start_index < len(sensor_df):
            window_df = sensor_df[start_index:start_index+window_rows]
            # Ensure that all windows are of the same length
            if len(window_df) == window_rows:
                # extra_timestamps = pd.date_range(
                #     start=window_df.index.max()+pd.Timedelta(milliseconds=1), 
                #     periods=window_rows-len(window_df), 
                #     freq='ms'
                # )
                # pd.concat([window_df, pd.DataFrame(np.nan, index=extra_timestamps, columns=window_df.columns)])
                windows.append(window_df)
            start_index = start_index + window_advance
    return windows


@app.cell
def _(cleaned, test_cleaned):
    fixed_windows_array = generate_fixed_windows(df=cleaned, sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP)
    test_fixed_windows_array = generate_fixed_windows(df=test_cleaned, sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP)
    print(len(fixed_windows_array))
    print(len(test_fixed_windows_array))
    return fixed_windows_array, test_fixed_windows_array


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define neural network.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prepare tensors
    """)
    return


@app.cell
def _(fixed_windows_array, test_fixed_windows_array):
    x_train_tensor = np.stack([df[AXES].values for df in fixed_windows_array], axis=0)
    y_train_tensor = np.array([df['label'].mode().iloc[0] for df in fixed_windows_array])
    x_test_tensor = np.stack([df[AXES].values for df in test_fixed_windows_array], axis=0)
    y_test_tensor = np.array([df['label'].mode().iloc[0] for df in test_fixed_windows_array])
    return x_test_tensor, x_train_tensor, y_test_tensor, y_train_tensor


@app.cell
def _(cleaned_unique_labels):
    simplest_CNN_model = models.Sequential([
        Input(INPUT_SHAPE),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=len(cleaned_unique_labels), activation='softmax')
    ])

    simplest_CNN_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    simplest_CNN_model.summary()
    return (simplest_CNN_model,)


@app.cell
def _(
    cleaned_unique_labels,
    simplest_CNN_model,
    x_test_tensor,
    x_train_tensor,
    y_test_tensor,
    y_train_tensor,
):
    _history = simplest_CNN_model.fit(x_train_tensor, label_encoder.transform(y_train_tensor), epochs=EPOCHS)
    _y_pred_probability = simplest_CNN_model.predict(x_test_tensor)
    _y_pred = label_encoder.inverse_transform(np.argmax(_y_pred_probability, axis=1))

    _precision_score = sm.precision_score(y_test_tensor, _y_pred, average=None)
    _recall_score = sm.recall_score(y_test_tensor, _y_pred, average=None)
    for _i, _label in enumerate(cleaned_unique_labels):
        print(f'For label {_label}: Precision {_precision_score[_i]:.2f}, Recall {_recall_score[_i]:.2f}')

    _cm = sm.confusion_matrix(y_test_tensor, _y_pred)
    _disp = sm.ConfusionMatrixDisplay(confusion_matrix=_cm, display_labels=label_encoder.classes_)
    _disp.plot(cmap=plt.cm.Blues)
    plt.title('Simplest CNN Confusion Matrix')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Improve
    """)
    return


@app.cell
def _(cleaned_unique_labels):
    cnn_model_improved = models.Sequential([
        Input(INPUT_SHAPE),
        Conv1D(filters=32, kernel_size=8, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.25),

        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.25),

        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling1D(), 
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(units=len(cleaned_unique_labels), activation='softmax')
    ])
    cnn_model_improved.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.AdamW(learning_rate=0.001), metrics=['accuracy'])
    cnn_model_improved.summary()
    return (cnn_model_improved,)


@app.cell
def _(
    cleaned_unique_labels,
    cnn_model_improved,
    x_test_tensor,
    x_train_tensor,
    y_test_tensor,
    y_train_tensor,
):
    _history = cnn_model_improved.fit(x_train_tensor, label_encoder.transform(y_train_tensor), epochs=EPOCHS)
    _y_pred_probability = cnn_model_improved.predict(x_test_tensor)
    _y_pred = label_encoder.inverse_transform(np.argmax(_y_pred_probability, axis=1))

    _precision_score = sm.precision_score(y_test_tensor, _y_pred, average=None)
    _recall_score = sm.recall_score(y_test_tensor, _y_pred, average=None)
    for _i, _label in enumerate(cleaned_unique_labels):
        print(f'For label {_label}: Precision {_precision_score[_i]:.2f}, Recall {_recall_score[_i]:.2f}')

    _cm = sm.confusion_matrix(y_test_tensor, _y_pred)
    _disp = sm.ConfusionMatrixDisplay(confusion_matrix=_cm, display_labels=label_encoder.classes_)
    _disp.plot(cmap=plt.cm.Blues)
    plt.title('Improved CNN Confusion Matrix')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try a hybrid CNN/RNN model.
    This should use convolutions to capture spatial features, and then use LSTM to capture temporal dependencies.
    """)
    return


@app.cell
def _(cleaned_unique_labels):
    cnn_rnn_model = models.Sequential([
        Input(INPUT_SHAPE),

        # CNN feature extractor.
        Conv1D(64, kernel_size=5, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling1D(2),

        Conv1D(128, kernel_size=5, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling1D(2),

        Conv1D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling1D(2),

        # Reshape for LSTM: [batch, reduced_timesteps, features]
        Reshape((-1, 128)),

        # Bidirectional LSTM.
        Bidirectional(LSTM(128, dropout=0.3, return_sequences=True)),
        Bidirectional(LSTM(64, dropout=0.3)),

        # Classification head.
        Dropout(0.5),
        Dense(len(cleaned_unique_labels), activation='softmax')
    ])

    # Usage
    cnn_rnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.AdamW(learning_rate=0.001), metrics=['accuracy'])
    cnn_rnn_model.summary()
    return (cnn_rnn_model,)


@app.cell
def _(
    cleaned_unique_labels,
    cnn_rnn_model,
    x_test_tensor,
    x_train_tensor,
    y_test_tensor,
    y_train_tensor,
):
    _history = cnn_rnn_model.fit(x_train_tensor, label_encoder.transform(y_train_tensor), epochs=EPOCHS)
    _y_pred_probability = cnn_rnn_model.predict(x_test_tensor)
    _y_pred = label_encoder.inverse_transform(np.argmax(_y_pred_probability, axis=1))

    _precision_score = sm.precision_score(y_test_tensor, _y_pred, average=None)
    _recall_score = sm.recall_score(y_test_tensor, _y_pred, average=None)
    for _i, _label in enumerate(cleaned_unique_labels):
        print(f'For label {_label}: Precision {_precision_score[_i]:.2f}, Recall {_recall_score[_i]:.2f}')

    _cm = sm.confusion_matrix(y_test_tensor, _y_pred)
    _disp = sm.ConfusionMatrixDisplay(confusion_matrix=_cm, display_labels=label_encoder.classes_)
    _disp.plot(cmap=plt.cm.Blues)
    plt.title('CNN-RNN Confusion Matrix')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
