import pandas as pd

def engineer_features(data, training_mode=True, mean_amount=None, mean_amount_by_hour=None):
    """
    Generates new features from the raw data.

    Args:
        data: The input DataFrame.
        training_mode: True if generating features for training data, False for test data.
        mean_amount: The mean transaction amount calculated from the training data (only used when training_mode is False)
        mean_amount_by_hour: The mean transaction amount per hour calculated from the training data (only used when training_mode is False)

    Returns:
        DataFrame with new features.
    """

    data['Hour'] = data['Time'] % 24

    if training_mode:
        mean_amount = data['Amount'].mean()
        mean_amount_by_hour = data.groupby('Hour')['Amount'].mean()

    data['Amount_relative_to_mean'] = data['Amount'] / mean_amount

    # Amount relative to the mean amount for the hour of the day
    if not training_mode:
        data['Amount_relative_to_mean_hour'] = data.apply(lambda row: row['Amount'] / mean_amount_by_hour.get(row['Hour'], 1), axis=1)
    else:
        data['Amount_relative_to_mean_hour'] = data.apply(lambda row: row['Amount'] / mean_amount_by_hour[row['Hour']], axis=1)

    # Example of lagged features (you'll likely need card IDs for more accurate lagged features)
    # This is a simplified example and may not be very effective without card IDs
    data.sort_values(['Time'], inplace=True)  # Sort by time
    data['Lagged_Amount'] = data['Amount'].shift(1)
    data['Amount_Diff'] = data['Amount'] - data['Lagged_Amount']
    data.fillna(0, inplace=True) # Fill NaN values after shift

    # More advanced features (would ideally require more data like user ID or card ID):
    # - Number of transactions in the last hour/day for the same user/card
    # - Average transaction amount for the same user/card in the last hour/day
    # - Time since the last transaction for the same user/card
    # - Whether the transaction occurred during a "risky" time period (e.g., late night)

    # Example of creating an interaction feature:
    data['Amount_x_Time'] = data['Amount'] * data['Time']

    return data, mean_amount, mean_amount_by_hour