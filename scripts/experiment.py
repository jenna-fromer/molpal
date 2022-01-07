from collections import Counter
import csv
import heapq
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np

from molpal.acquirer import metrics

Point = Tuple[str, float]


class Experiment:
    """An Experiment represents the output of a MolPAL run

    It can be queried for the progress at a given iteration, the order in which
    points were acquired, and other conveniences
    """

    def __init__(self, experiment: Union[Path, str], d_smi_idx: Optional[Dict] = None):
        self.experiment = Path(experiment)
        self.d_smi_idx = d_smi_idx

        try:
            chkpts_dir = self.experiment / "chkpts"
            chkpt_dirs = [
                d for d in chkpts_dir.iterdir() if len(d.stem.split("_")) <= 2
            ]
            self.chkpt_dirs = sorted(chkpt_dirs, key=lambda d: int(d.stem.split("_")[1]))
            config = Experiment.read_config(self.experiment / "config.ini")
            try:
                self.k = int(config["top-k"])
            except ValueError:
                self.k = int(float(config["top-k"]) * len(d_smi_idx))
            self.metric = config["metric"]
            self.beta = float(config.get("beta", 2.0))
            self.xi = float(config.get("xi", 0.001))
            
            self.new_style = True
        except FileNotFoundError:
            self.new_style = False

        data_dir = self.experiment / "data"
        try:
            self.__size = len(Experiment.read_scores(
                data_dir / "all_explored_final.csv"
            ))
        except:
            self.__size = None
            pass
        scores_csvs = [p for p in data_dir.iterdir() if "final" not in p.stem]
        self.scores_csvs = sorted(scores_csvs, key=lambda p: int(p.stem.split("_")[-1]))
        self.__sizes = [len(self[i]) for i in range(self.num_iters)]

    def __len__(self) -> int:
        """the total number of inputs sampled in this experiment"""
        if self.__size is None:
            raise IncompleteExperimentError
            
        return self.__size

    def __getitem__(self, i: int) -> Dict:
        """Get the score data for iteration i, where i=0 is the initialization batch"""
        return Experiment.read_scores(self.scores_csvs[i])

    def __iter__(self) -> Iterable[Dict]:
        """iterate through all the score data at each iteration"""
        for scores_csv in self.scores_csvs:
            yield dict(Experiment.read_scores(scores_csv))

    @property
    def num_iters(self) -> int:
        """the total number of iterations in this experiment, including the
        initialization batch"""
        return len(self.scores_csvs)

    @property
    def init_size(self) -> int:
        """the size of this experiment's initialization batch"""
        return self.__sizes[0]

    @property
    def num_acquired(self) -> List[int]:
        """The total number of points acquired *by* iteration i, where i=0 is the
        initialization batch"""
        return self.__sizes

    def get(self, i: int, N: Optional[int] = None) -> Dict:
        """get the top-N scores for iteration i"""
        return dict(Experiment.read_scores(self.scores_csvs[i], N))

    def new_points_by_epoch(self) -> List[Dict]:
        """get the set of new points acquired at each iteration in the list of
        scores_csvs that are already sorted by iteration"""
        new_points_by_epoch = []
        all_points = {}

        for scores in self:
            new_points = {
                smi: score for smi, score in scores.items() if smi not in all_points
            }
            new_points_by_epoch.append(new_points)
            all_points.update(new_points)

        return new_points_by_epoch

    def predictions(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """get the predictions for exploration iteration i.

        The exploration iterations are 1-indexed to account for the
        iteration of the initialization batch. So self.predictions[1] corresponds to the first exploration iteteration

        Returns
        -------
        means : np.ndarray
        vars : np.ndarray

        Raises
        ------
        ValueError
            if i is less than 1
        """
        if i not in range(1, self.num_iters):
            raise ValueError(f"arg: i must be in {{1..{self.num_iters}}}. got {i}")
        preds_npz = np.load(self.chkpt_dirs[i] / "preds.npz")

        return preds_npz["Y_pred"], preds_npz["Y_var"]

    def utilities(self) -> List[np.ndarray]:
        if not self.new_style:
            raise NotImplementedError(
                "Utilities cannot be calculated for an old style MolPAL run"
            )
        Us = []

        for i in range(1, self.num_iters):
            Y_pred, Y_var = self.predictions(i)
            ys = list(self[i - 1].values())

            Y = np.nan_to_num(np.array(ys, dtype=float), nan=-np.inf)
            current_max = np.partition(Y, -self.k)[-self.k]

            Us.append(
                metrics.calc(
                    self.metric,
                    Y_pred,
                    Y_var,
                    current_max,
                    0.0,
                    self.beta,
                    self.xi,
                    False,
                )
            )

        return Us

    def points_in_order(self) -> List[Point]:
        """Get all points acquired during this experiment's run in the order
        in which they were acquired"""
        if self.d_smi_idx is None:
            raise NotImplementedError(
                'Cannot get points in order without setting "self.d_smi_idx"'
            )
        if not self.new_style:
            raise NotImplementedError(
                "Cannot get points in order for an old style MolPAL run"
            )
        init_batch, *exp_batches = self.new_points_by_epoch()

        all_points_in_order = []
        all_points_in_order.extend(init_batch.items())

        for new_points, U in zip(exp_batches, self.utilities()):
            us = np.array([U[self.d_smi_idx[smi]] for smi in new_points])

            new_points_in_order = [
                smi_score
                for _, smi_score in sorted(zip(us, new_points.items()), reverse=True)
            ]
            all_points_in_order.extend(new_points_in_order)

        return all_points_in_order

    def reward_curve(self, true_top_k: List[Point], reward: str = "scores"):
        """Calculate the reward curve of a molpal run

        Parameters
        ----------
        true_top_k : List
            the list of the true top-k molecules as tuples of their SMILES string
            and associated score
        reward : str, default='scores'
            the type of reward to calculate

        Returns
        -------
        np.ndarray
            the reward as a function of the number of molecules sampled
        """
        all_points_in_order = self.points_in_order()
        k = len(true_top_k)

        if reward == "scores":
            _, true_scores = zip(*true_top_k)
            missed_scores = Counter(true_scores)

            all_hits_in_order = np.zeros(len(all_points_in_order), dtype=bool)
            for i, (_, score) in enumerate(all_points_in_order):
                if score not in missed_scores:
                    continue
                all_hits_in_order[i] = True
                missed_scores[score] -= 1
                if missed_scores[score] == 0:
                    del missed_scores[score]
            reward_curve = 100 * np.cumsum(all_hits_in_order) / k

        elif reward == "smis":
            true_top_k_smis = {smi for smi, _ in true_top_k}
            all_hits_in_order = np.array(
                [smi in true_top_k_smis for smi, _ in all_points_in_order], dtype=bool
            )
            reward_curve = 100 * np.cumsum(all_hits_in_order) / k

        elif reward == "top-k-ave":
            reward_curve = np.zeros(len(all_points_in_order), dtype="f8")
            heap = []

            for i, (_, score) in enumerate(all_points_in_order[:k]):
                if score is not None:
                    heapq.heappush(heap, score)
                top_k_avg = sum(heap) / k
                reward_curve[i] = top_k_avg
            reward_curve[:k] = top_k_avg

            for i, (_, score) in enumerate(all_points_in_order[k:]):
                if score is not None:
                    heapq.heappushpop(heap, score)

                top_k_avg = sum(heap) / k
                reward_curve[i + k] = top_k_avg

        elif reward == "total-ave":
            _, all_scores_in_order = zip(*all_points_in_order)
            Y = np.array(all_scores_in_order, dtype=float)
            Y = np.nan_to_num(Y)
            N = np.arange(0, len(Y)) + 1
            reward_curve = np.cumsum(Y) / N

        else:
            raise ValueError

        return reward_curve

    def calculate_reward(
        self,
        i: int,
        true: List[Point],
        is_sorted: bool = False,
        maximize: bool = True,
        avg: bool = True,
        smis: bool = True,
        scores: bool = True,
    ) -> Tuple[float, float, float]:
        """calculate the reward for iteration i

        Parameters
        ----------
        i : int
            the iteration to calculate the reward for
        true : List[Point]
            the true top-N points
        is_sorted : bool, default=True
            whether the true points are sorted by objective value=
        avg : bool, default=True
            whether to calculate the average reward=
        smis : bool, default=True
            whether to calcualte the SMILES reward=
        scores : bool, default=True
            whether to calcualte the scores reward

        Returns
        -------
        f_avg : float
            the fraction of the true top-k average score
        f_smis : float
            the fraction of the true top-k SMILES identified
        f_scores : float
            the fraction of the true top-k score identified
        """
        N = len(true)
        if not is_sorted:
            true = sorted(true, key=lambda kv: kv[1], reverse=maximize)

        found = list(self.get(i, N).items())

        found_smis, found_scores = zip(*found)
        true_smis, true_scores = zip(*true)

        if avg:
            found_avg = np.mean(found_scores)
            true_avg = np.mean(true_scores)
            f_avg = found_avg / true_avg
        else:
            f_avg = None

        if smis:
            found_smis = set(found_smis)
            true_smis = set(true_smis)
            correct_smis = len(found_smis & true_smis)
            f_smis = correct_smis / len(true_smis)
        else:
            f_smis = None

        if scores:
            missed_scores = Counter(true_scores)
            missed_scores.subtract(found_scores)
            n_missed_scores = sum(
                count if count > 0 else 0 for count in missed_scores.values()
            )
            f_scores = (N - n_missed_scores) / N
        else:
            f_scores = None

        return f_avg, f_smis, f_scores

    def calculate_cluster_fraction(
        self, i: int, true_clusters: Tuple[Set, Set, Set]
    ) -> Tuple[float, float, float]:
        large, mids, singletons = true_clusters
        N = len(large) + len(mids) + len(singletons)

        found = set(list(self.get(i, N).keys()))

        f_large = len(found & large) / len(large)
        f_mids = len(found & mids) / len(mids)
        f_singletons = len(found & singletons) / len(singletons)

        return f_large, f_mids, f_singletons

    @staticmethod
    def read_scores(
        scores_csv: Union[Path, str], N: Optional[int] = None
    ) -> List[Tuple]:
        """read the scores contained in the file located at scores_csv"""
        with open(scores_csv) as fid:
            reader = csv.reader(fid)
            next(reader)

            smis_scores = [
                (row[0], float(row[1])) if row[1] else (row[0], -float("inf"))
                for row in reader
            ]

        if N is None:
            return smis_scores

        return sorted(smis_scores, key=lambda xy: xy[1], reverse=True)[:N]

    @staticmethod
    def read_config(config_file: str) -> Dict:
        """parse an autogenerated MolPAL config file to a dictionary"""
        with open(config_file) as fid:
            return dict(line.split(" = ") for line in fid.read().splitlines())

    @staticmethod
    def boltzmann(xs: Iterable[float]) -> float:
        X = np.array(xs)
        E = np.exp(-X)
        Z = E.sum()
        return (X * E / Z).sum()

class IncompleteExperimentError(Exception):
    pass

if __name__ == "__main__":
    pass