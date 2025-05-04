from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


# ------------- TODO: Implement the following environment -------------
class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        horizon: int = 10,
    ):
        """Initializes the observation and action space for the environment."""
        self.horizon = int(horizon)

        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Returns
        -------
        obs : int
            Initial state (always 0).
        info : dict
            An empty info dictionary.
        """
        self.current_steps = 0
        self.position = 0
        return self.position, {}

    def step(
        self, action: int
    ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take one step in the environment.

        Parameters
        ----------
        action : int
            Action to take (0: move to state 0, 1: move to state 1).

        Returns
        -------
        next_obs : int
            The resulting state.
        reward : float
            The reward at the new state.
        terminated : bool
            Whether the episode ended due to task success (always False here).
        truncated : bool
            Whether the episode ended due to reaching the time limit.
        info : dict
            An empty dictionary.
        """
        action = int(action)
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        self.current_steps += 1

        self.position = action

        reward = float(action)
        terminated = False
        truncated = self.current_steps >= self.horizon

        return self.position, reward, terminated, truncated, {}

    def get_reward_per_action(self) -> np.ndarray:
        """
        Return the reward function R[s, a] for each (state, action) pair.

        R[s, a] is the reward after taking action a in state s.

        Returns
        -------
        R : np.ndarray
            A (n_states, n_actions) array of rewards.
        """
        nS, nA = self.observation_space.n, self.action_space.n
        R = np.zeros((nS, nA), dtype=float)
        for s in range(nS):
            for a in range(nA):
                R[s, a] = float(a)
        return R

    def get_transition_matrix(
        self,
    ) -> np.ndarray:
        """
        Construct a deterministic transition matrix T[s, a, s'].

        Returns
        -------
        T : np.ndarray
            A (num_states, num_actions, num_states) tensor where
            T[s, a, s'] = probability of transitioning to s' from s via a.
        """
        nS, nA = 2, 2
        T = np.zeros((nS, nA, nS), dtype=float)
        for s in range(0, 1):
            for a in range(0, 1):
                s_next = a
                T[s, a, s_next] = float(1)
        return T


class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: gym.Env, noise: float = 0.1, seed: int | None = None):
        super().__init__(env)
        assert 0.0 <= noise <= 1.0, "noise must be in [0,1]"
        self.noise = noise
        self.rng = np.random.default_rng(seed)

        self.obserservation_space = env.observation_space
        self.action_space = env.action_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """
        Reset the base environment and return a noisy observation.

        Parameters
        ----------
        seed : int or None, optional
            Seed for the reset, by default None.
        options : dict or None, optional
            Additional reset options, by default None.

        Returns
        -------
        obs : int
            The (possibly noisy) initial observation.
        info : dict
            Additional info returned by the environment.
        """
        true_obs, info = self.env.reset(seed=seed, options=options)
        return self._noisy_obs(true_obs), info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment and return a noisy observation.

        Parameters
        ----------
        action : int
            Action to take.

        Returns
        -------
        obs : int
            The (possibly noisy) resulting observation.
        reward : float
            The reward received.
        terminated : bool
            Whether the episode terminated.
        truncated : bool
            Whether the episode was truncated due to time limit.
        info : dict
            Additional information from the base environment.
        """
        true_obs, reward, terminated, truncated, info = self.env.step(action)
        return self._noisy_obs(true_obs), reward, terminated, truncated, info

    def _noisy_obs(self, true_obs: int) -> int:
        """
        Return a possibly noisy version of the true observation.

        With probability `noise`, replaces the true observation with
        a randomly selected incorrect state.

        Parameters
        ----------
        true_obs : int
            The true observation/state index.

        Returns
        -------
        obs : int
            A noisy (or true) observation.
        """
        if self.rng.random() < self.noise:
            n = self.observation_space.n
            others = [s for s in range(n) if s != true_obs]
            return int(self.rng.choice(others))
        else:
            return int(true_obs)

    def render(self, mode: str = "human"):
        """
        Render the current state of the environment.

        Parameters
        ----------
        mode : str, optional
            Render mode, by default "human".

        Returns
        -------
        Any
            Rendered output from the base environment.
        """
        return self.env.render(mode=mode)
