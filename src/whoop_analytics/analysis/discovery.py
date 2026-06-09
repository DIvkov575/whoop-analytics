from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


@dataclass(frozen=True)
class CausalLink:
    source: str
    target: str
    lag: int
    strength: float
    p_value: float


@dataclass(frozen=True)
class DiscoveryResult:
    links: list[CausalLink]
    variable_names: list[str]
    val_matrix: np.ndarray
    p_matrix: np.ndarray
    graph: np.ndarray


@dataclass
class CausalDiscovery:
    max_lag: int = 3
    significance_level: float = 0.05

    def run(self, df: pd.DataFrame, target: str) -> DiscoveryResult:
        variable_names = list(df.columns)
        target_idx = variable_names.index(target)

        data_array = df.values.astype(np.float64)
        dataframe = pp.DataFrame(data_array, var_names=variable_names)

        parcorr = ParCorr(significance="analytic")
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

        results = pcmci.run_pcmci(tau_max=self.max_lag, pc_alpha=None)

        val_matrix = results["val_matrix"]
        p_matrix = results["p_matrix"]
        graph = pcmci.get_graph_from_pmatrix(
            p_matrix=p_matrix,
            alpha_level=self.significance_level,
            tau_min=1,
            tau_max=self.max_lag,
        )

        links = self._extract_links(
            val_matrix=val_matrix,
            p_matrix=p_matrix,
            graph=graph,
            variable_names=variable_names,
            target_idx=target_idx,
        )

        return DiscoveryResult(
            links=links,
            variable_names=variable_names,
            val_matrix=val_matrix,
            p_matrix=p_matrix,
            graph=graph,
        )

    def _extract_links(
        self,
        val_matrix: np.ndarray,
        p_matrix: np.ndarray,
        graph: np.ndarray,
        variable_names: list[str],
        target_idx: int,
    ) -> list[CausalLink]:
        links = []
        n_vars = len(variable_names)

        for source_idx in range(n_vars):
            for lag in range(1, self.max_lag + 1):
                if graph[source_idx, target_idx, lag] == "-->":
                    links.append(CausalLink(
                        source=variable_names[source_idx],
                        target=variable_names[target_idx],
                        lag=lag,
                        strength=float(val_matrix[source_idx, target_idx, lag]),
                        p_value=float(p_matrix[source_idx, target_idx, lag]),
                    ))

        links.sort(key=lambda l: abs(l.strength), reverse=True)
        return links
