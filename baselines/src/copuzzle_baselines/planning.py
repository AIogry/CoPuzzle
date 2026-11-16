"""Bounded best-first search using only a supplied learned transition predictor."""
from __future__ import annotations

import heapq

import numpy as np

from .config import integer


class ModelPlanner:
    """A state-mismatch heuristic; no puzzle-specific validity or distance oracle.

    The predictor is called through the evaluator's budget for every attempted
    hypothetical transition. Learned illegal edges and self-loops are skipped.
    Tie rotation is within-case memory; the model parameters never change.
    """
    def __init__(self, predictor, n_actions, *, batch_predictor=None, beam_width=32, max_depth=8):
        self.predictor = predictor
        self.batch_predictor = batch_predictor
        self.n_actions = integer(n_actions, "n_actions", 1)
        self.beam_width = integer(beam_width, "beam_width", 1)
        self.max_depth = integer(max_depth, "max_depth", 1)
        self.decisions = 0

    def _expand(self, board, order, budget):
        if self.batch_predictor is not None and hasattr(budget, "predict_batch"):
            actions = np.asarray(order[:min(self.n_actions, budget.limit - budget.used)], dtype=np.int32)
            if not len(actions):
                return
            boards = np.repeat(np.asarray(board, dtype=np.int32)[None], len(actions), axis=0)
            predictions, legal = budget.predict_batch(self.batch_predictor, boards, actions)
            predictions, legal = np.asarray(predictions), np.asarray(legal)
            if predictions.shape != boards.shape or legal.shape != actions.shape:
                raise ValueError("model batch has the wrong shape")
            if predictions.dtype.kind not in "iub" or legal.dtype.kind not in "iub" or np.any((legal != 0) & (legal != 1)):
                raise ValueError("model outputs must be integer boards and binary legality")
            for a, following, flag in zip(actions, predictions, legal):
                yield int(a), following, bool(flag)
        else:
            for action in order:
                if budget.used == budget.limit:
                    break
                following, flag = budget.predict(self.predictor, np.asarray(board, dtype=np.int32), action)
                following, flag = np.asarray(following), np.asarray(flag)
                if following.shape != (len(board),) or following.dtype.kind not in "iub" or flag.shape != () or flag.dtype.kind not in "iub" or flag not in (0, 1):
                    raise ValueError("model outputs must be an integer board and binary legality")
                yield action, following, bool(flag)

    def __call__(self, observation, budget):
        if budget.limit <= 0:
            raise ValueError("model-only planning requires a positive model-query budget; it has no zero-query actor")
        start = tuple(np.asarray(observation["board"]).tolist())
        goal = tuple(np.asarray(observation["goal"]).tolist())
        depth_limit = min(self.max_depth, observation["steps_remaining"])
        order = [(i + self.decisions) % self.n_actions for i in range(self.n_actions)]
        self.decisions += 1
        frontier = [(sum(x != y for x, y in zip(start, goal)), 0, 0, start, -1)]
        visited = {start: 0}
        serial = 0
        fallback, best = order[0], (float("inf"), float("inf"))
        while frontier and budget.used < budget.limit:
            _, depth, _, board, first = heapq.heappop(frontier)
            if depth >= depth_limit or visited[board] < depth:
                continue
            for action, prediction, legal in self._expand(board, order, budget):
                if not legal:
                    continue
                following = tuple(np.asarray(prediction).tolist())
                if len(following) != len(start):
                    raise ValueError("model prediction has the wrong width")
                if following == board:
                    continue
                initial = action if first < 0 else first
                if following == goal:
                    return initial
                mismatch = sum(x != y for x, y in zip(following, goal))
                if (mismatch, depth + 1) < best:
                    best, fallback = (mismatch, depth + 1), initial
                if visited.get(following, depth + 2) <= depth + 1:
                    continue
                visited[following] = depth + 1
                serial += 1
                heapq.heappush(frontier, (mismatch + depth + 1, depth + 1, serial, following, initial))
            if len(frontier) > self.beam_width:
                frontier = heapq.nsmallest(self.beam_width, frontier)
                heapq.heapify(frontier)
        return fallback
