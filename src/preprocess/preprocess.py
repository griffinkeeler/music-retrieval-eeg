from sklearn.preprocessing import RobustScaler


def robust_scale_channels(raw):
    """Legacy per-recording robust scaling; not used by window generation.

    This fits on the complete Raw recording and is not split-aware. New EEG
    windows use a fixed volts-to-microvolts conversion instead.
    """
    data = raw.get_data()
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data.T).T
    raw._data = data_scaled

    return raw
