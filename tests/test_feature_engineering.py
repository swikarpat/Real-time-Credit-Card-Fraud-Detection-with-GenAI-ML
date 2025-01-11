import unittest
import pandas as pd
from src.feature_engineering import engineer_features

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # Create a small dummy DataFrame for testing
        data = {'Time': [1, 2, 3, 4, 5, 6],
                'Amount': [10, 20, 30, 40, 50, 60]}
        self.test_df = pd.DataFrame(data)

    def test_engineer_features_training(self):
        # Test feature engineering in training mode
        df_train, mean_amount, mean_amount_by_hour = engineer_features(self.test_df.copy(), training_mode=True)

        self.assertIsNotNone(mean_amount)
        self.assertIsNotNone(mean_amount_by_hour)
        self.assertTrue('Hour' in df_train.columns)
        self.assertTrue('Amount_relative_to_mean' in df_train.columns)
        self.assertTrue('Amount_relative_to_mean_hour' in df_train.columns)
        self.assertTrue('Lagged_Amount' in df_train.columns)
        self.assertTrue('Amount_Diff' in df_train.columns)
        self.assertTrue('Amount_x_Time' in df_train.columns)

        # Add specific assertions to check the values of the engineered features
        # For example:
        self.assertEqual(df_train['Hour'].iloc[0], 1)
        self.assertAlmostEqual(df_train['Amount_relative_to_mean'].iloc[0], 0.2857, places=4)

    def test_engineer_features_testing(self):
        # Test feature engineering in testing mode
        mean_amount = 35  # Example mean from training data
        mean_amount_by_hour = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60} # Example hourly means from training data
        df_test, _, _ = engineer_features(self.test_df.copy(), training_mode=False, mean_amount=mean_amount, mean_amount_by_hour=mean_amount_by_hour)

        self.assertTrue('Hour' in df_test.columns)
        self.assertTrue('Amount_relative_to_mean' in df_test.columns)
        self.assertTrue('Amount_relative_to_mean_hour' in df_test.columns)
        self.assertTrue('Lagged_Amount' in df_test.columns)
        self.assertTrue('Amount_Diff' in df_test.columns)
        self.assertTrue('Amount_x_Time' in df_test.columns)

        # Add specific assertions
        self.assertAlmostEqual(df_test['Amount_relative_to_mean'].iloc[0], 0.2857, places=4)
        self.assertAlmostEqual(df_test['Amount_relative_to_mean_hour'].iloc[0], 1.0)
        self.assertEqual(df_test['Amount_Diff'].iloc[0], 0)
        self.assertEqual(df_test['Amount_x_Time'].iloc[0], 10)

if __name__ == '__main__':
    unittest.main()