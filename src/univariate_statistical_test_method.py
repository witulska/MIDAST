from typing import Union
import numpy as np
import pandas as pd
from tqdm import trange

from scipy.stats import (
    ks_2samp,
    anderson_ksamp,
    cramervonmises_2samp,
    epps_singleton_2samp,
)


class ChangeDetector1D:
    def __init__(
        self,
        test_name: str = "KS",
    ) -> None:
        """
        Change point detection for univariate data.

        Available tests:
        - "KS"  : Kolmogorov–Smirnov
        - "AD"  : Anderson–Darling k-sample
        - "CVM" : Cramér–von Mises
        - "ES"  : Epps–Singleton
        """
        self.test_name = test_name
        self.window_size = None
        self.results_df = None

    def _run_test(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """
        Run selected two-sample test.

        Returns:
            (pvalue, statistic)
        """
        if self.test_name == "KS":
            res = ks_2samp(x, y)
            return res.pvalue, res.statistic

        elif self.test_name == "AD":
            res = anderson_ksamp([x, y])
            # scipy zwraca significance_level w %
            pvalue = res.significance_level / 100
            return pvalue, res.statistic

        elif self.test_name == "CVM":
            res = cramervonmises_2samp(x, y)
            return res.pvalue, res.statistic

        elif self.test_name == "ES":
            res = epps_singleton_2samp(x, y)
            return res.pvalue, res.statistic

        else:
            raise NotImplementedError(f"Unknown test: {self.test_name}")

    def test_in_window(
        self,
        data: np.ndarray,
        window_size: int,
        shift: int = 10,
    ) -> pd.DataFrame:
        """
        Apply two-sample tests on sliding windows.
        """
        if data.ndim != 1:
            raise ValueError("Input data must be one-dimensional.")

        results_stat = {}
        results_pval = {}

        n = len(data)

        for ind in range(0, n - 2 * window_size, shift):
            w1 = data[ind : ind + window_size]
            w2 = data[ind + window_size : ind + 2 * window_size]

            pval, stat = self._run_test(w1, w2)

            center = ind + window_size
            results_stat[center] = stat
            results_pval[center] = pval

        df = pd.DataFrame(
            {
                "id": results_stat.keys(),
                "window1_start": [i - window_size for i in results_stat],
                "window2_end": [i + window_size for i in results_stat],
                "statistic": results_stat.values(),
                "pvalue": results_pval.values(),
            }
        )

        return df

    def fit(
        self,
        data: Union[pd.Series, np.ndarray],
        window_size: int,
        shift: int = 10,
    ) -> pd.DataFrame:
        """
        Run change detection.
        """
        if isinstance(data, pd.Series):
            data = data.values

        self.window_size = window_size
        self.results_df = self.test_in_window(
            data=data,
            window_size=window_size,
            shift=shift,
        )
        return self.results_df

    def analyze_results(
        self,
        results_df: pd.DataFrame,
        alpha: float = 0.05,
        shift_group: int | None = None,
        max_no_changes: int | None = None,
        based_on: str = "statistic",
        output_type: str = "np.array",
    ) -> Union[np.ndarray, pd.DataFrame, None]:
        """
        Group and select change points.
        """
        cp = results_df[results_df.pvalue <= alpha]

        if cp.empty:
            return None if output_type == "np.array" else pd.DataFrame()

        if shift_group is None:
            shift_group = self.window_size

        cp = cp.sort_values("id").reset_index(drop=True)
        cp["group"] = 0

        group = 1
        cp.loc[0, "group"] = group

        for i in range(1, len(cp)):
            if cp.loc[i, "id"] - cp.loc[i - 1, "id"] <= shift_group:
                cp.loc[i, "group"] = group
            else:
                group += 1
                cp.loc[i, "group"] = group

        if based_on == "statistic":
            selected = (
                cp.loc[cp.groupby("group")["statistic"].idxmax()]
                .sort_values("statistic", ascending=False)
            )
        elif based_on == "pvalue":
            selected = (
                cp.loc[cp.groupby("group")["pvalue"].idxmin()]
                .sort_values("pvalue")
            )
        else:
            raise ValueError("based_on must be 'statistic' or 'pvalue'")

        if max_no_changes:
            selected = selected.head(max_no_changes)

        if output_type == "pd.DataFrame":
            return selected
        elif output_type == "np.array":
            return selected.id.values
        else:
            raise ValueError("output_type must be 'np.array' or 'pd.DataFrame'")
