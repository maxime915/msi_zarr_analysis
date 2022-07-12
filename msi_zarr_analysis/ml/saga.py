"""SAGA: Simulated Annealing and Genetic Algorithms

Module for feature selection, see saga() function.

Some references:
    - https://doi.org/10.1016/j.patcog.2009.06.009
    - https://cse.iitkgp.ac.in/~dsamanta/courses/sca/resources/slides/GA-03%20Selection%20Strategies.pdf
    - https://uomustansiriyah.edu.iq/media/lectures/6/6_2021_11_29!08_04_39_AM.pdf
"""

import functools
import logging
import time
from collections import deque
from typing import Any, NamedTuple, Tuple

import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score


def saga(
    model: BaseEstimator,
    time_budget: float,
    data: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray = None,
    cv=None,
    rng=None,
    low_variance_threshold: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Perform the SAGA algorithm for `time_budget` seconds to find an optimal
    feature selection for the  classification problem `(data, target)|,
    evaluated by cross validation of `model` (with `cv` strategy, taking
    `groups` into account). For reproducibility, `rng` may be set to a fixed seed.

    SAGA: Simulated Annealing and Genetic Algorithms.
    REF: https://doi.org/10.1016/j.patcog.2009.06.009

    Args:
        model (BaseEstimator): Estimator to use for the prediction
        time_budget (float): Time (seconds) the algorithm should run for
        data (np.ndarray): Inputs (n_samples, n_features)
        target (np.ndarray): Targets (n_samples, n_outputs) or (n_samples,)
        groups (np.ndarray, optional): Grouping of samples (n_samples,). Defaults to None.
        cv (Any, optional): Folding strategy used to evaluate the score, \
            may be an int or KFold or GroupKFold, etc. \
            NOTE: To have group cross validation, it is **necessary** that cv \
            is either a GroupKFold or StratifiedGroupKFold instance, passing groups \
            is not sufficient. Defaults to None (scikit-learn's default).
        rng (Any, optional): Random number generator. Defaults to None.
        low_variance_threshold (float, optional): remove features for which the variance \
            is <= low_variance_threshold. A 0 value only remove constant features, \
            a negative value won't remove any feature. Defaults to 0.0.

    Returns:
        Tuple[np.ndarray, float]: best found solution (n_features,) and its score
    """

    # store which feature will be selected
    solution = np.zeros(data.shape[1:], dtype=bool)

    # remove features which have a variance <=
    if low_variance_threshold >= 0:
        pre_selection = np.var(data, axis=0) > low_variance_threshold
        logging.debug("removed feature before SAGA: %s", np.argwhere(~pre_selection))
        data = data[:, pre_selection]

    # scorer to assess the fitness of a feature selection solution
    scorer = Scorer(model, data, target, groups, cv)

    # Phase 1 : explore the search space to find multiple local minima
    population, scores = simulated_annealing(
        scorer, 0.5 * time_budget, population_size=100, rng=rng
    )

    # Phase 2 : refine minima to find better values
    population, scores = genetic_algorithms(
        scorer, population, scores, 0.3 * time_budget, rng
    )

    # Phase 3 : fine-tune the best solution to date
    individual, score = hill_climbing(
        scorer, population, scores, 0.2 * time_budget, 10000, rng
    )

    # report the solution with the right shape (i.e. including removed features)
    if low_variance_threshold >= 0:
        solution[pre_selection] = individual
    else:
        solution = individual

    return solution, score


class Array(NamedTuple):
    "hashable wrapper around a numpy array"

    data: np.ndarray

    def __eq__(self, other: "Array") -> bool:
        return np.array_equal(self.data, other.data)

    def __hash__(self) -> int:
        return hash(self.data.tobytes())


def score_dataset(model, data: np.ndarray, target: np.ndarray, groups: np.ndarray, cv):
    scores = cross_val_score(model, data, target, groups=groups, cv=cv, n_jobs=-1)
    return np.mean(scores)


class Scorer(NamedTuple):
    "Object scoring feature selection solutions"

    model: BaseEstimator
    data: np.ndarray
    target: np.ndarray
    groups: np.ndarray
    cv: Any
    
    def __eq__(self, __o: object) -> bool:
        return id(self) == id(__o)

    def __hash__(self) -> int:
        return id(self)

    @functools.lru_cache(maxsize=2**14)
    def __score_feature_selection_cached(self, array: Array) -> float:
        data_ = self.data[:, array.data]
        return score_dataset(self.model, data_, self.target, self.groups, self.cv)

    def score_feature_selection(self, array: np.ndarray) -> float:
        return self.__score_feature_selection_cached(Array(array))


def simulated_annealing(
    scorer: Scorer,
    time_budget: float,
    population_size: int,
    rng=None,
) -> Tuple[np.ndarray, np.ndarray]:
    "performs the simulated annealing to find good feature selection solutions"

    rng = _make_rng(rng)

    def get_scores(population_: np.ndarray):
        start = time.time()
        scores = np.zeros((population_.shape[0],))
        for idx, individual in enumerate(population_):
            scores[idx] = scorer.score_feature_selection(individual)
        end = time.time()
        return scores, end - start

    # Step 2.
    temp_init = time_budget
    temp_curr = temp_init

    # Step 3. Individuals drawn i.i.d. , with each feature having p=0.5 of being selected
    population = rng.random(size=(population_size, scorer.data.shape[1])) < 0.5
    # Step 4.
    scores, duration = get_scores(population)
    # Step 5.
    temp_curr -= duration

    _log_stats(scores, "SA")

    while temp_curr > 0:

        # Step 6. random permutations
        prob_mutation = 0.5 * (1 - np.exp(-temp_curr / temp_init))
        flip_mask = rng.random(size=population.shape) < prob_mutation

        # Step 7. evaluate permutations
        next_population = np.logical_xor(flip_mask, population)
        next_scores, duration = get_scores(next_population)

        # Step 8. decide which permutation to accept
        prob_acceptance = np.exp((scores - next_scores) / (-temp_curr))
        acceptance = rng.random(size=scores.shape) < prob_acceptance

        population[acceptance, :] = next_population[acceptance, :]
        scores[acceptance] = next_scores[acceptance]

        _log_stats(scores, "SA")

        # Step 9. update temperature
        temp_curr -= duration

    return population, scores


def genetic_algorithms(
    scorer: Scorer,
    start_population: np.ndarray,
    start_scores: np.ndarray,
    time_budget: float,
    rng=None,
) -> Tuple[np.ndarray, np.ndarray]:
    "perform genetic algorithms to combine good solutions of feature selection"

    rng = _make_rng(rng)

    assert start_population.shape[0] % 2 == 0

    def draw_individual(scores_: np.ndarray):
        "selection by ranking"

        ranking = rankdata(scores_, method="min")
        probabilities = ranking / ranking.sum()

        return rng.choice(
            scores_.shape[0], size=(population.shape[0],), p=probabilities
        )

    def hux(left: np.ndarray, right: np.ndarray):
        # compute the number of swaps to perform
        difference = left != right
        hamming_distances = np.sum(difference)

        # randomly (p=0.5) swap the mismatching elements
        swap_probabilities = rng.random(size=(left.shape[0],))
        swap_mask = difference  # only consider mismatching elements
        swap_mask[swap_probabilities < 0.5] = False  # randomly remove swaps

        old_sum = np.sum(swap_mask)

        # limit the number of swaps
        swap_counts = np.cumsum(swap_mask)
        max_idx = np.searchsorted(swap_counts, hamming_distances / 2, side="right")
        swap_mask[max_idx:] = False

        assert np.sum(swap_mask) in [np.floor(hamming_distances / 2), old_sum]

        return np.where(swap_mask, left, right), np.where(swap_mask, right, left)

    def update_population(population_: np.ndarray, scores_: np.ndarray):
        "crossover and mutation"

        # Step 2.
        parent_indices = draw_individual(scores_)
        new_population = np.zeros_like(population_)

        # Step 3, 4 : crossover
        index = 0
        for left, right in zip(parent_indices[::2], parent_indices[1::2]):
            children = hux(population_[left], population_[right])
            new_population[index : index + 2] = children
            index += 2

        # Step 5 : mutation
        mutation_mask = rng.random(size=new_population.shape) < 0.001
        new_population = np.logical_xor(mutation_mask, new_population)

        # Step 6 : update scores
        scores_ = np.zeros_like(scores_)
        for idx, individual in enumerate(new_population):
            scores_[idx] = scorer.score_feature_selection(individual)

        return new_population, scores_

    # Step 1.
    population = start_population
    scores = start_scores

    _log_stats(scores, "GA")

    # repeat the update process until the time budget is exhausted
    while time_budget > 0:
        start = time.time()
        population, scores = update_population(population, scores)
        end = time.time()

        _log_stats(scores, "GA")

        # Step 7.
        time_budget -= end - start

    return population, scores


def hill_climbing(
    scorer: Scorer,
    start_population: np.ndarray,
    start_scores: np.ndarray,
    time_budget: float,
    branching_factor: int,
    rng: np.random.Generator = None,
):
    "fine-tune the best solution of feature selection"

    rng = _make_rng(rng)

    best_idx = np.argmax(start_scores)

    # Step 1.
    best_score = start_scores[best_idx]
    best_individual = start_population[best_idx]

    queue = deque([(best_individual, best_score)])

    while time_budget > 0:
        start = time.time()

        individual, score = queue.popleft()

        _log_stats(best_score, "HC")

        for _ in range(branching_factor):

            # Step 2.
            trial = individual.copy()
            mutate_idx = rng.integers(trial.size)
            trial[mutate_idx] = not trial[mutate_idx]

            # Step 3.
            trial_score = scorer.score_feature_selection(trial)

            # Step 4.
            if trial_score > score:
                queue.append((trial, trial_score))

                # Step 5.
                if trial_score > best_score:
                    best_score = trial_score
                    best_individual = trial

                break
        else:
            break

        end = time.time()
        time_budget -= end - start

    _log_stats(best_score, "HC")

    return best_individual, best_score


def _make_rng(rng):
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _log_stats(scores, parent: str):
    if not isinstance(scores, np.ndarray):
        logging.debug("[%s] score = %.2f", parent, scores)
        return

    logging.debug(
        "[%s] scores = %.2f pm %.2f ; in [%.2f, %.2f]",
        parent,
        np.mean(scores),
        np.std(scores),
        np.min(scores),
        np.max(scores),
    )
