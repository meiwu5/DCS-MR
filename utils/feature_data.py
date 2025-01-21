import numpy as np

def add_time_features(data, add_time_in_day=True, add_day_in_week=True ):
    data = np.expand_dims(data, axis=-1)
    print('input_data.shape:',data.shape)
    num_samples, num_nodes, num_features = data.shape
    feature_list = [data]  # Start with the original data.

    if add_time_in_day:
        time_ind = [i % 288 / 288 for i in range(num_samples)]  # Normalize time index to [0, 1].
        time_ind = np.array(time_ind)
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))  # Expand to match data shape.
        feature_list.append(time_in_day)

    if add_day_in_week:
        day_in_week = [(i // 288) % 7 for i in range(num_samples)]  # Compute day index (0-6).
        day_in_week = np.array(day_in_week)
        day_in_week = np.tile(day_in_week, [1, num_nodes, 1]).transpose((2, 1, 0))  # Expand to match data shape.
        feature_list.append(day_in_week)

    data_with_features = np.concatenate(feature_list, axis=-1)

    print('data_with_features.shape:',data_with_features.shape)

    return data_with_features
