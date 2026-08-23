import enum
import math
import typing
from collections import deque

import numpy as np
from gymnasium import Env
from gymnasium import spaces
from enum import Enum

from ._vendor import cellworld_game as cwgame
from .utils import normalize_angle

# Fields that get frame-stacked (capture temporal dynamics)
STACK_FIELDS = [
    "prey_x",
    "prey_y",
    "prey_direction",
    "predator_visible",
    "predator_x",
    "predator_y",
    "predator_direction",
    "time_prey_seen_predator",
]


# ---------------------------------------------------------------------------
# Shared base classes (mirrors botevade_gym.py style)
# ---------------------------------------------------------------------------

class Observation(np.ndarray):
    fields = []

    def __init__(self):
        super().__init__()
        for index, field in enumerate(self.__class__.fields):
            self._create_property(index=index, field=field)
        self.field_enum = Enum("fields", {f: i for i, f in enumerate(self.__class__.fields)})

    def __new__(cls):
        shape = (len(cls.fields),)
        obj = super(Observation, cls).__new__(cls, shape, np.float32, None, 0, None, None)
        obj.fill(0)
        return obj

    def _create_property(self, index: int, field: str):
        def getter(self):
            return self[index]
        def setter(self, value):
            self[index] = value
        setattr(self.__class__, field, property(getter, setter))

    def __setitem__(self, field, value):
        if isinstance(field, Enum):
            np.ndarray.__setitem__(self, field.value, value)
        else:
            np.ndarray.__setitem__(self, field, value)

    def __getitem__(self, field):
        if isinstance(field, Enum):
            return np.ndarray.__getitem__(self, field.value)
        return np.ndarray.__getitem__(self, field)


class Environment(Env):
    def __init__(self):
        self.event_handlers: typing.Dict[str, typing.List[typing.Callable]] = {
            "reset": [], "step": []
        }

    def __handle_event__(self, event_name: str, *args):
        for handler in self.event_handlers[event_name]:
            handler(*args)

    def add_event_handler(self, event_name: str, handler: typing.Callable):
        if event_name not in self.event_handlers:
            raise ValueError(f"Unknown event: {event_name}")
        self.event_handlers[event_name].append(handler)

    def reset(self, options=None, seed=None):
        Env.reset(self, seed=seed)
        self.__handle_event__("reset", options, seed)

    def step(self, action):
        self.__handle_event__("step", action)


# ---------------------------------------------------------------------------
# Observation definition
# ---------------------------------------------------------------------------

class OasisObservation(Observation):
    fields = [
        # --- stackable (temporal context) ---
        "prey_x",
        "prey_y",
        "prey_direction",
        "predator_visible",
        "predator_x",
        "predator_y",
        "predator_direction",
        "time_prey_seen_predator",
        # --- non-stack (current frame only) ---
        "puffed",
        "puff_cooled_down",
        "finished",
        "prey_goal_distance",
        "goal_x",           # current active goal x
        "goal_y",           # current active goal y
        "goals_remaining",  # how many goals are still in the queue
    ]


# ---------------------------------------------------------------------------
# Main Gymnasium environment
# ---------------------------------------------------------------------------

class OasisEnv(Environment):

    metadata = {"render_modes": []}

    PointOfView = cwgame.Oasis.PointOfView
    AgentRenderMode = cwgame.Agent.RenderMode

    class ObservationType(enum.Enum):
        DATA = 0
        PIXELS = 1

    class ActionType(enum.Enum):
        DISCRETE = 0
        CONTINUOUS = 1

    # Default goal locations for world "21_05"
    DEFAULT_GOAL_LOCATIONS = [
        (0.265625, 0.5),
        (0.3125, 0.7435696448143734),
        (0.3125, 0.1752404735808355),
        (0.4765625, 0.45940505919760444),
        (0.640625, 0.7435696448143734),
        (0.6875, 0.1752404735808355),
        (0.78125, 0.5),
    ]

    def __init__(self,
                 world_name: str = "21_05",
                 goal_locations: typing.List[typing.Tuple[float, float]] = None,
                 use_predator: bool = True,
                 use_lppos: bool = False,
                 max_step: int = 500,
                 reward_function: typing.Callable[["OasisObservation"], float] = lambda obs: 0.0,
                 time_step: float = 0.25,
                 model_time_step: float = 0.025,
                 render: bool = False,
                 real_time: bool = False,
                 point_of_view: "OasisEnv.PointOfView" = None,
                 agent_render_mode: "OasisEnv.AgentRenderMode" = None,
                 observation_type: "OasisEnv.ObservationType" = None,
                 action_type: "OasisEnv.ActionType" = None,
                 frame_stack_k: int = 3,
                 puff_cool_down_time: float = 0.5,
                 puff_threshold: float = 0.1,
                 goal_threshold: float = 0.025,
                 goal_time: float = 1.0,
                 max_line_of_sight_distance: float = 1.0,
                 prey_max_forward_speed: float = 0.5,
                 prey_max_turning_speed: float = 20.0,
                 predator_prey_forward_speed_ratio: float = 0.15,
                 predator_prey_turning_speed_ratio: float = 0.175):

        # resolve defaults that can't be in the signature (mutable enum references)
        if point_of_view is None:
            point_of_view = OasisEnv.PointOfView.TOP
        if agent_render_mode is None:
            agent_render_mode = OasisEnv.AgentRenderMode.SPRITE
        if observation_type is None:
            observation_type = OasisEnv.ObservationType.DATA
        if action_type is None:
            action_type = OasisEnv.ActionType.DISCRETE

        if observation_type == OasisEnv.ObservationType.PIXELS and not render:
            raise ValueError("PIXELS observation type requires render=True")

        if goal_locations is None:
            goal_locations = OasisEnv.DEFAULT_GOAL_LOCATIONS

        self.max_step = max_step
        self.reward_function = reward_function
        self.time_step = time_step
        self.action_type = action_type
        self.observation_type = observation_type
        self.frame_stack_k = frame_stack_k

        # Loader for action list
        self.loader = cwgame.CellWorldLoader(world_name=world_name)
        if use_lppos:
            self.action_list = self.loader.tlppo_action_list
        else:
            self.action_list = self.loader.full_action_list

        # Action space
        if self.action_type == OasisEnv.ActionType.DISCRETE:
            self.action_space = spaces.Discrete(len(self.action_list))
        else:
            # Continuous: (x, y) destination in [0, 1]
            self.action_space = spaces.Box(0.0, 1.0, (2,), dtype=np.float32)

        # Build the underlying Oasis model.
        # model_time_step is the physics integration step (fine-grained, e.g. 0.025).
        # time_step (gym level) controls how much simulated time passes per gym step()
        # call — the gym loop runs (time_step / model_time_step) model steps each time.
        self.model = cwgame.Oasis(
            world_name=world_name,
            goal_locations=goal_locations,
            use_predator=use_predator,
            puff_cool_down_time=puff_cool_down_time,
            puff_threshold=puff_threshold,
            goal_threshold=goal_threshold,
            goal_time=goal_time,
            time_step=model_time_step,   # ← physics step, NOT gym step
            real_time=real_time,
            render=render,
            point_of_view=point_of_view,
            agent_render_mode=agent_render_mode,
            max_line_of_sight_distance=max_line_of_sight_distance,
        )

        # Set agent speeds (Oasis doesn't expose these in its constructor,
        # but NavigationAgent stores them as plain attributes used in step()).
        self.model.prey.max_forward_speed = prey_max_forward_speed
        self.model.prey.max_turning_speed = prey_max_turning_speed
        if use_predator:
            self.model.predator.max_forward_speed = prey_max_forward_speed * predator_prey_forward_speed_ratio
            self.model.predator.max_turning_speed = prey_max_turning_speed * predator_prey_turning_speed_ratio

        # Observation space setup
        if self.observation_type == OasisEnv.ObservationType.DATA:
            self.observation = OasisObservation()
            self.stack_indices = [
                self.observation.fields.index(f) for f in STACK_FIELDS
            ]
            self.nonstack_indices = [
                i for i in range(self.observation.shape[0]) if i not in self.stack_indices
            ]
            stacked_shape = (
                len(self.stack_indices) * self.frame_stack_k + len(self.nonstack_indices),
            )
            self.observation_space = spaces.Box(-np.inf, np.inf, stacked_shape, dtype=np.float32)
            self.frame_stack = deque(maxlen=self.frame_stack_k)
        else:
            self.observation = self.model.view.get_screen(normalized=True)
            self.observation_space = spaces.Box(0.0, 1.0, self.observation.shape, dtype=np.float32)
            self.frame_stack = None

        # Episode trackers
        self.step_count = 0
        self.episode_reward = 0.0
        self.time_prey_seen_predator = -1
        self._prev_puffed = False

        Environment.__init__(self)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predator_visible(self) -> bool:
        """Check if the predator is within line-of-sight of the prey."""
        if not self.model.use_predator:
            return False
        return self.model.visibility.line_of_sight(
            self.model.prey.state.location,
            self.model.predator.state.location,
        )

    def __update_observation__(self):
        if self.observation_type == OasisEnv.ObservationType.DATA:
            obs = self.observation

            # Prey state
            obs.prey_x = self.model.prey.state.location[0]
            obs.prey_y = self.model.prey.state.location[1]
            obs.prey_direction = normalize_angle(math.radians(self.model.prey.state.direction))

            # Predator state (only if visible)
            if self._predator_visible():
                obs.predator_visible = 1.0
                obs.predator_x = self.model.predator.state.location[0]
                obs.predator_y = self.model.predator.state.location[1]
                obs.predator_direction = normalize_angle(
                    math.radians(self.model.predator.state.direction)
                )
                self.time_prey_seen_predator = self.step_count
            else:
                obs.predator_visible = 0.0
                obs.predator_x = 0.0
                obs.predator_y = 0.0
                obs.predator_direction = 0.0

            obs.time_prey_seen_predator = self.time_prey_seen_predator

            # Puff state (one-shot flag — cleared after reading)
            obs.puffed = float(self.model.puffed)
            obs.puff_cooled_down = self.model.puff_cool_down

            # Episode / goal state
            obs.finished = float(not self.model.running)
            obs.prey_goal_distance = self.model.prey_goal_distance
            if self.model.goal_location is not None:
                obs.goal_x = self.model.goal_location[0]
                obs.goal_y = self.model.goal_location[1]
            else:
                obs.goal_x = 0.0
                obs.goal_y = 0.0
            obs.goals_remaining = float(len(self.model.goal_sequence))
        else:
            self.observation = self.model.view.get_screen(normalized=True)

        return self.__get_stacked_observation__()

    def __get_stacked_observation__(self):
        if self.observation_type != OasisEnv.ObservationType.DATA:
            return self.observation
        current_obs = np.array(self.observation, copy=True)
        current_stack = current_obs[self.stack_indices]
        current_nonstack = current_obs[self.nonstack_indices]
        self.frame_stack.append(current_stack)
        while len(self.frame_stack) < self.frame_stack_k:
            self.frame_stack.appendleft(np.zeros_like(current_stack))
        stacked = np.concatenate(list(self.frame_stack), axis=0)
        return np.concatenate([stacked, current_nonstack], axis=0)

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def set_action(self, action: typing.Union[int, typing.Tuple[float, float]]):
        if self.action_type == OasisEnv.ActionType.DISCRETE:
            self.model.prey.set_destination(self.action_list[action])
        else:
            self.model.prey.set_destination(tuple(action[:2]))

    # ------------------------------------------------------------------
    # Core gym API
    # ------------------------------------------------------------------

    def __step__(self):
        self.step_count += 1
        truncated = self.step_count >= self.max_step
        obs = self.__update_observation__()
        reward = self.reward_function(obs)
        self.episode_reward += reward

        # Clear one-shot puff flag after it has been read into the observation
        if self.model.puffed:
            self.model.puffed = False

        done = not self.model.running
        if done or truncated:
            survived = int(done and self.model.puff_count == 0)
            info = {
                "puff_count": self.model.puff_count,
                "episode_reward": self.episode_reward,
                "is_success": survived,
                "survived": survived,
            }
        else:
            info = {}

        return obs, reward, done, truncated, info

    def step(self, action: typing.Union[int, typing.Tuple[float, float]]):
        self.set_action(action=action)
        model_t = self.model.time + self.time_step
        while self.model.running and self.model.time < model_t:
            self.model.step()
        Environment.step(self, action=action)
        return self.__step__()

    def __reset__(self):
        self.step_count = 0
        self.episode_reward = 0.0
        self.time_prey_seen_predator = -1
        obs = self.__update_observation__()
        if self.observation_type == OasisEnv.ObservationType.DATA:
            self.frame_stack.clear()
            current_obs = np.array(self.observation, copy=True)
            current_stack = current_obs[self.stack_indices]
            for _ in range(self.frame_stack_k):
                self.frame_stack.append(np.array(current_stack, copy=True))
            obs = self.__get_stacked_observation__()
        return obs, {}

    def reset(self, options=None, seed=None):
        self.model.reset()
        Environment.reset(self, options=options, seed=seed)
        return self.__reset__()

    def close(self):
        self.model.close()
        Env.close(self=self)
