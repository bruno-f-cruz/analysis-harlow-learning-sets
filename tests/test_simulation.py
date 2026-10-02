import numpy as np
import pandas as pd

from analysis.features import label_first_stop
from analysis.simulation import trials_from_rollout


def test_first_stop_label_matches_label_first_stop():
    rng = np.random.default_rng(0)
    n_animals, n_blocks, block_len = 3, 40, 10
    block = np.tile(np.repeat(np.arange(n_blocks), block_len), n_animals)
    subject = np.repeat(np.arange(n_animals), n_blocks * block_len)
    trial = np.tile(np.arange(n_blocks * block_len), n_animals)
    stay = rng.random(len(block)) < 0.3  # some blocks end up with no stay at all
    df = pd.DataFrame(
        {
            "subject_id": subject,
            "block": block,
            "trial": trial,
            "stay": stay.astype(float),
            "reward": (stay & (rng.random(len(block)) < 0.5)).astype(float),
            "odor_rewarded": rng.random(len(block)) < 0.5,
        }
    )
    trials = trials_from_rollout(df)
    expected = label_first_stop(trials.drop(columns="first_stop_rewarded"))
    assert trials["first_stop_rewarded"].isna().any()  # exercises the no-stop blocks
    pd.testing.assert_series_equal(
        trials["first_stop_rewarded"], expected["first_stop_rewarded"]
    )
