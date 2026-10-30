"""Small WorldLedger imagination component, with DreamerV3 return semantics.

lambda_return is a NumPy adaptation of danijar/dreamerv3 agent.py at
e3f02248693a79dc8b0ebd62c93683888ddaccfe, MIT, Copyright (c) 2024 Danijar
Hafner. Full notice: licenses/DreamerV3-MIT.txt.
This module does not implement DreamerV3's RSSM or actor-critic training.
"""
from __future__ import annotations

import copy
import numpy as np


def lambda_return(last, term, rew, val, boot, disc=0.98, lam=0.95):
    """State-aligned [batch, H+1] inputs; reward at t+1 belongs to action t.

    Terminal stops bootstrap; last marks a sequence boundary and stops the
    recursive trace while retaining the next-state bootstrap. val is retained
    for compatibility with the upstream interface and shape contract.
    """
    last, term, rew, val, boot = [np.asarray(x, dtype=np.float64)
                                 for x in (last, term, rew, val, boot)]
    if rew.ndim != 2 or rew.shape[1] < 2:
        raise ValueError("expected batch x (horizon+1)")
    if any(x.shape != rew.shape or not np.isfinite(x).all()
           for x in (last, term, rew, val, boot)):
        raise ValueError("inconsistent shapes or nonfinite arrays")
    if not 0 <= disc <= 1 or not 0 <= lam <= 1:
        raise ValueError("invalid discount or lambda")
    if any(np.any((x < 0) | (x > 1)) for x in (last, term)):
        raise ValueError("boundary/terminal values must lie in [0,1]")
    rets = [boot[:, -1]]
    live = (1 - term)[:, 1:] * disc
    cont = (1 - last)[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return np.stack(list(reversed(rets))[:-1], 1)


def imagine_plans(candidate, particles, plans, *, gamma=0.98, lam=1.0, value_fn=None):
    """Evaluate each fixed open-loop plan in one coherent candidate world.

    Each particle follows the same externally selected action sequence. Future
    real observations/states and an oracle are absent from this API. value_fn,
    if provided, must be a separately trained candidate-local value estimate.
    The development default uses zero bootstrap and lambda=1 (finite returns).
    """
    if len(particles) == 0 or len(plans) == 0 or len(plans[0]) == 0:
        raise ValueError("nonempty particles and plans required")
    horizon = len(plans[0])
    if any(len(p) != horizon for p in plans):
        raise ValueError("inconsistent plan lengths")
    values = []
    transitions = 0
    model = candidate.components
    for plan in plans:
        shape = (len(particles), horizon + 1)
        rewards, terminal, last, bootstrap = [np.zeros(shape) for _ in range(4)]
        last[:, -1] = 1
        for i, particle in enumerate(particles):
            state = copy.deepcopy(particle)
            done = False
            for t, action in enumerate(plan):
                if done:
                    terminal[i, t + 1] = 1
                    continue
                nxt = model['transition_model'](copy.deepcopy(state), int(action))
                reward, done = model['reward_model'](state, int(action), nxt)
                rewards[i, t + 1] = float(reward)
                terminal[i, t + 1] = float(bool(done))
                bootstrap[i, t + 1] = 0 if value_fn is None or done else float(value_fn(nxt))
                state = nxt
                transitions += 1
        values.append(lambda_return(last, terminal, rewards, bootstrap, bootstrap,
                                    gamma, lam)[:, 0])
    result = np.asarray(values)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite imagined returns")
    return {"plan_particle_returns": result,
            "plan_mean_returns": result.mean(axis=1),
            "model_transition_calls": transitions}


def categorical_cem(score_fn, *, actions, horizon, population=128,
                    iterations=4, elites=16, smoothing=0.25,
                    probability_floor=0.05, seed=0):
    """Optimize a fixed-length discrete action sequence with categorical CEM.

    ``score_fn`` receives an integer array of shape ``[population, horizon]``
    and returns one finite score per plan. Higher is better. The caller owns
    world-model rollouts and their accounting, which keeps this optimizer usable
    for matched single-world and ensemble comparisons.
    """
    actions = np.asarray(tuple(actions), dtype=np.int64)
    if actions.ndim != 1 or len(actions) < 2 or len(np.unique(actions)) != len(actions):
        raise ValueError("actions must contain at least two unique values")
    if min(int(horizon), int(population), int(iterations), int(elites)) <= 0:
        raise ValueError("horizon, population, iterations, and elites must be positive")
    if elites > population:
        raise ValueError("elites cannot exceed population")
    if not 0 <= smoothing <= 1:
        raise ValueError("smoothing must lie in [0,1]")
    if not 0 <= probability_floor < 1 / len(actions):
        raise ValueError("probability_floor must lie in [0, 1/num_actions)")
    rng = np.random.default_rng(int(seed))
    probabilities = np.full((int(horizon), len(actions)), 1 / len(actions))
    best_plan, best_score = None, -np.inf
    evaluated = 0
    for _ in range(int(iterations)):
        indices = np.column_stack([
            rng.choice(len(actions), size=int(population), p=row)
            for row in probabilities
        ])
        plans = actions[indices]
        scores = np.asarray(score_fn(plans), dtype=np.float64)
        if scores.shape != (int(population),) or not np.isfinite(scores).all():
            raise ValueError("score_fn must return one finite score per plan")
        evaluated += len(scores)
        winner = int(np.argmax(scores))
        if scores[winner] > best_score:
            best_score = float(scores[winner])
            best_plan = plans[winner].copy()
        elite_indices = np.argsort(-scores, kind="stable")[:int(elites)]
        frequencies = np.zeros_like(probabilities)
        for t in range(int(horizon)):
            frequencies[t] = np.bincount(indices[elite_indices, t],
                                          minlength=len(actions)) / int(elites)
        probabilities = smoothing * probabilities + (1 - smoothing) * frequencies
        probabilities = ((1 - len(actions) * probability_floor) * probabilities
                         + probability_floor)
    return {"best_plan": best_plan.tolist(), "best_score": best_score,
            "evaluated_plans": evaluated,
            "action_probabilities": probabilities}
