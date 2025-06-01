import pandas as pd
from timeseries_toolkit import TimeSeriesSuitabilityScorer


def test_scorer():
    scorer = TimeSeriesSuitabilityScorer(min_observations=10)
    df = pd.read_csv("./data/bitcoin.csv")
    result = scorer.score(df, date_column='timeOpen', value_columns=["open","high","low","close","volume","marketCap"])
    print(f"Score: {result['overall_score']:.1f}/100")
    print(f"Suitable: {result['is_timeseries_suitable']}")
    print("Issues:", result['issues'])