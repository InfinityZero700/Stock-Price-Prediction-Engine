import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from stock_predictor import forecast_prices, load_stock_data


class StockPredictorTests(unittest.TestCase):
    def test_forecast_length(self):
        params, fitted, future = forecast_prices(np.array([10, 11, 12, 13, 14]), 10)
        self.assertEqual(len(future), 10)
        self.assertEqual(len(fitted), 4)
        self.assertTrue(np.isfinite(future).all())

    def test_csv_formats_and_chronological_order(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'prices.csv'
            pd.DataFrame({
                'Date': ['2026-01-07', '2026-01-05', '2026-01-06'],
                'Open': ['$12', '$10', '$11'],
                'High': ['$13', '$11', '$12'],
                'Low': ['$11', '$9', '$10'],
                'Close/Last': ['$12', '$10', '$11'],
                'Volume': ['1,200', '1,000', '1,100'],
            }).to_csv(path, index=False)
            loaded = load_stock_data(path)
            self.assertEqual(loaded['AvgPrice'].tolist(), [10, 11, 12])
            self.assertEqual(loaded['Volume'].tolist(), [1000, 1100, 1200])


if __name__ == '__main__':
    unittest.main()
