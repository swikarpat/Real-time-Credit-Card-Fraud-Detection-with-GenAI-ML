import unittest
import pandas as pd
from src.data_ingestion import load_data
import os
import yaml

class TestDataIngestion(unittest.TestCase):
    def setUp(self):
        # Create a dummy config file
        self.config_content = """
        train_test_split:
            test_size: 0.2
            random_state: 42
        """
        with open("test_config.yaml", "w") as f:
            yaml.dump(yaml.safe_load(self.config_content), f)

        # Create a small dummy CSV for testing
        data = {'Time': [3600, 7200, 10800], 'Amount': [10, 20, 30], 'Class': [0, 1, 0]}
        self.test_df = pd.DataFrame(data)
        self.test_df.to_csv("test_data.csv", index=False)

    def tearDown(self):
        # Clean up the dummy files
        os.remove("test_data.csv")
        os.remove("test_config.yaml")

    def test_load_data(self):
        train_data, test_data = load_data("test_data.csv", "test_config.yaml")

        # Assertions to check if data is loaded and processed correctly
        self.assertIsNotNone(train_data)
        self.assertIsNotNone(test_data)
        self.assertEqual(len(train_data), 2)
        self.assertEqual(len(test_data), 1)
        self.assertTrue('Time' in train_data.columns)
        self.assertTrue('Amount' in train_data.columns)
        self.assertTrue('Class' in train_data.columns)

if __name__ == '__main__':
    unittest.main()