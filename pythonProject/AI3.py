import os
import sys
import time
import random
import json
import csv
import warnings
from collections import deque, namedtuple
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
CONFIG_PATH = r"C:\Users\Sara\Desktop\Traffic_RL_Project\pythonProject\config.json"

#IF DON'T FIND JSON, USE THIS
DEFAULTS = {
    "SUMO_CONFIG": r"D:\YYY.sumocfg",
    "USE_GUI": True,
    "MAX_TRAIN_STEPS": 10000,
    "TRAIN_START": 2000,
    "BUFFER_CAPACITY": 50000,
    "BATCH_SIZE": 64,
    "GAMMA": 0.99,
    "LR": 1e-4,
    "TARGET_UPDATE_FREQ": 2000, #STEPS FOR SYNCING WITH NET WEIGHTS
    "EPS_START": 1.0,
    "EPS_END": 0.05,
    "EPS_DECAY": 100000.0,
    "MIN_GREEN": 5,
    "SWITCH_PENALTY": 0.25, #BEFORE GREEN
    "QUEUE_NORM": 20.0,
    "USE_SPEED_IN_STATE": True,
    "SPEED_NORM": 13.89,
    "USE_WAITING_TIME_IN_REWARD": True,
    "REWARD_WEIGHTS": {"halts": 1.0, "queue": 0.5, "waiting": 0.75, "switch_penalty": 0.25},
    "EVAL_EVERY": 2500,
    "EVAL_EPISODES": 3,
    "EPISODE_LENGTH": 1200,
    "CHECKPOINT_DIR": "checkpoints",
    "LOG_CSV": "training_log.csv",
    "SEED": 42, #MAKES RANDOM NUMS FOR SAME RESUTS
    "RESUME_TRAINING": False,
    "RESUME_CHECKPOINT": None,
    "USE_SAFE_SWITCH": True,
    "SAFE_SWITCH_DISTANCE": 10.0,
    "BASELINE_PHASE_DURATION": 20,
    "PLOT_DIR": "plots",
    "SAVE_REPLAY": False,
    "SAVE_REPLAY_MAX": 100000
}

#READING JSON
def load_config_from_json(path: str, defaults: dict) -> dict:
    cfg = defaults.copy()
    if not os.path.exists(path):
        print(f"[config] file not found at: {path} — using defaults.")
        return cfg
    try:
        with open(path, 'r', encoding='utf-8') as f:
            user_cfg = json.load(f)
        if not isinstance(user_cfg, dict):
            print(f"[config] invalid format in {path}, must be JSON object. Using defaults.")
            return cfg
        for k, v in user_cfg.items():
            if k in defaults:
                cfg[k] = v
            else:
                print(f"[config] unknown key in config.json: {k} (ignored)")
        print(f"[config] loaded from {path}")
    except Exception as e:
        print(f"[config] failed to load {path}: {e} — using defaults.")
    return cfg

# LOAD CONFIG AND DEFAULTS BY DICTIONARY
CFG = load_config_from_json(CONFIG_PATH, DEFAULTS)

#PARAMETERS
SUMO_CONFIG = CFG["SUMO_CONFIG"]
USE_GUI = CFG["USE_GUI"]
MAX_TRAIN_STEPS = int(CFG["MAX_TRAIN_STEPS"])
TRAIN_START = int(CFG["TRAIN_START"])
BUFFER_CAPACITY = int(CFG["BUFFER_CAPACITY"])
BATCH_SIZE = int(CFG["BATCH_SIZE"])
GAMMA = float(CFG["GAMMA"])
LR = float(CFG["LR"])
TARGET_UPDATE_FREQ = int(CFG["TARGET_UPDATE_FREQ"])
EPS_START = float(CFG["EPS_START"])
EPS_END = float(CFG["EPS_END"])
EPS_DECAY = float(CFG["EPS_DECAY"])
MIN_GREEN = int(CFG["MIN_GREEN"])
SWITCH_PENALTY = float(CFG["SWITCH_PENALTY"])
QUEUE_NORM = float(CFG["QUEUE_NORM"])
USE_SPEED_IN_STATE = bool(CFG["USE_SPEED_IN_STATE"])
SPEED_NORM = float(CFG["SPEED_NORM"])
USE_WAITING_TIME_IN_REWARD = bool(CFG["USE_WAITING_TIME_IN_REWARD"])
REWARD_WEIGHTS = CFG["REWARD_WEIGHTS"]
EVAL_EVERY = int(CFG["EVAL_EVERY"])
EVAL_EPISODES = int(CFG["EVAL_EPISODES"])
EPISODE_LENGTH = int(CFG["EPISODE_LENGTH"])
CHECKPOINT_DIR = CFG["CHECKPOINT_DIR"]
LOG_CSV = CFG["LOG_CSV"]
SEED = int(CFG["SEED"])
RESUME_TRAINING = bool(CFG["RESUME_TRAINING"])
RESUME_CHECKPOINT = CFG["RESUME_CHECKPOINT"]
USE_SAFE_SWITCH = bool(CFG["USE_SAFE_SWITCH"])
SAFE_SWITCH_DISTANCE = float(CFG["SAFE_SWITCH_DISTANCE"])
BASELINE_PHASE_DURATION = int(CFG["BASELINE_PHASE_DURATION"])
PLOT_DIR = CFG["PLOT_DIR"]
SAVE_REPLAY = bool(CFG.get("SAVE_REPLAY", False))
SAVE_REPLAY_MAX = int(CFG.get("SAVE_REPLAY_MAX", 100000))


os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

#SAME RANDOM NUMBERS FOR EACH RUN
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available(): #GPU
    torch.cuda.manual_seed(SEED)

warnings.filterwarnings("ignore", message=".*getAllProgramLogics.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)


def ensure_sumo_tools():
    if "SUMO_HOME" not in os.environ:
        candidates = [
            r"C:\Program Files (x86)\Eclipse\Sumo",
            r"C:\Program Files\Eclipse\Sumo",
            "/usr/share/sumo",
            "/opt/sumo"
        ]
        for c in candidates:
            if os.path.isdir(os.path.join(c, "tools")):
                os.environ["SUMO_HOME"] = c
                break
    if "SUMO_HOME" not in os.environ:
        sys.exit("SUMO_HOME is not set.")
    tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)

ensure_sumo_tools()
from sumolib import checkBinary
import traci
from traci import TraCIException


def resolve_config_path(p: str) -> str:
    p = os.path.abspath(os.path.expanduser(os.path.expandvars(p)))
    if not os.path.exists(p) or not p.lower().endswith(".sumocfg"):
        raise FileNotFoundError(f".sumocfg not found: {p}")
    return p

def select_green_phases(tls_id: str, verbose: bool = False) -> List[int]:
    actions = []
    try:
        logics = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
        phases = logics[0].phases if logics else []
        for i, ph in enumerate(phases):
            s = ph.state
            has_green = ("G" in s) or ("g" in s)
            all_red = all(ch.lower() == 'r' for ch in s)
            if has_green and (not all_red):
                actions.append(i)
        if verbose:
            print(f"[select_green_phases] tls={tls_id}, total_phases={len(phases)}, selected={actions}")
    except Exception as e:
        print(f"Error in select_green_phases: {e}")
    if not actions and 'phases' in locals() and phases:
        actions = list(range(len(phases)))
    return actions if actions else [0]

def save_config(config: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

class TrafficEnv:
    def __init__(self, sumo_cfg: str, use_gui: bool = False, max_steps: int = EPISODE_LENGTH,
                 min_green: int = MIN_GREEN, queue_norm: float = QUEUE_NORM, use_speed: bool = USE_SPEED_IN_STATE,
                 safe_switch: bool = USE_SAFE_SWITCH, safe_distance: float = SAFE_SWITCH_DISTANCE):
        self.sumo_cfg = resolve_config_path(sumo_cfg)
        self.use_gui = use_gui
        self.max_steps = int(max_steps)
        self.min_green = int(min_green)
        self.queue_norm = float(queue_norm)
        self.use_speed = use_speed
        self.sumo_binary = checkBinary("sumo-gui" if self.use_gui else "sumo")
        self.tls_id: Optional[str] = None
        self.allowed_phases: List[int] = []
        self.controlled_lanes: List[str] = []
        self.step_count = 0
        self.current_phase_index = 0
        self.elapsed_phase_time = 0
        self.phase_defs = []
        self.lane_lengths = {}
        self.action_dim = None
        self.state_dim = None
        self.safe_switch = bool(safe_switch)
        self.safe_distance = float(safe_distance)

    def _start_sumo(self):
        cmd = [
            self.sumo_binary, "-c", self.sumo_cfg,
            "--no-step-log", "true",
            "--duration-log.disable", "true",
            "--step-length", "1.0",
            "--seed", str(SEED)
        ]
        try:
            traci.start(cmd)
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            raise

    def _close_sumo(self):
        try:
            if traci.isLoaded():
                traci.close(False)
        except Exception:
            pass
        time.sleep(0.12)
#SRART 1 EPISODE
    def reset(self, verbose: bool = False) -> np.ndarray:
        try:
            if traci.isLoaded():
                traci.close(False)
                time.sleep(0.12)
        except Exception:
            pass

        self._start_sumo()
        self.step_count = 0
        self.elapsed_phase_time = 0

        tls_list = traci.trafficlight.getIDList()
        if not tls_list:
            self._close_sumo()
            raise RuntimeError("No traffic light found in SUMO network.")
        self.tls_id = tls_list[0]

        try:
            logics = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)
            if logics and len(logics) > 0:
                self.phase_defs = logics[0].phases
        except Exception:
            self.phase_defs = []

        self.allowed_phases = select_green_phases(self.tls_id)
        if not self.allowed_phases:
            self.allowed_phases = [0]
        self.current_phase_index = 0
        traci.trafficlight.setPhase(self.tls_id, int(self.allowed_phases[self.current_phase_index]))
        self.elapsed_phase_time = 0

        lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        seen = set(); uniq = []
        for l in lanes:
            if l not in seen:
                seen.add(l); uniq.append(l)
        self.controlled_lanes = uniq

        self.lane_lengths = {}
        for l in self.controlled_lanes:
            try:
                self.lane_lengths[l] = float(traci.lane.getLength(l))
            except Exception:
                self.lane_lengths[l] = 999.0

        lane_count = len(self.controlled_lanes)
        self.action_dim = len(self.allowed_phases)
        self.state_dim = lane_count * (1 + (1 if self.use_speed else 0)) + self.action_dim + 1

        if verbose:
            print(f"Environment initialized: {lane_count} lanes, {self.action_dim} actions, state_dim={self.state_dim}")

        return self._get_state()

#GET STATE AND GIVE IT TO NET
    def _get_state(self) -> np.ndarray:
        try:
            queues = [traci.lane.getLastStepHaltingNumber(l) for l in self.controlled_lanes]
            queues_arr = np.array(queues, dtype=np.float32) / (self.queue_norm if self.queue_norm > 0 else 1.0)

            parts = [queues_arr]

            if self.use_speed:
                speeds = [traci.lane.getLastStepMeanSpeed(l) for l in self.controlled_lanes]
                speeds_arr = np.array(speeds, dtype=np.float32) / (SPEED_NORM if SPEED_NORM > 0 else 1.0)
                parts.append(speeds_arr)

            onehot = np.zeros(self.action_dim, dtype=np.float32)
            onehot[self.current_phase_index] = 1.0
            t_norm = np.array([self.elapsed_phase_time / 30.0], dtype=np.float32)  # normalize

            parts.append(onehot); parts.append(t_norm)
            state = np.concatenate(parts, axis=0)

            if state.shape[0] != self.state_dim:
                desired = self.state_dim
                if state.shape[0] < desired:
                    pad = np.zeros(desired - state.shape[0], dtype=np.float32)
                    state = np.concatenate([state, pad], axis=0)
                else:
                    state = state[:desired]

            return state
        except Exception as e:
            print(f"Error in _get_state: {e}")
            return np.zeros(self.state_dim if self.state_dim else 1, dtype=np.float32)

    def _compute_reward(self) -> float:
        try:
            halts = sum(int(traci.lane.getLastStepHaltingNumber(l)) for l in self.controlled_lanes)
            queue_avg = (halts / max(1, len(self.controlled_lanes)))
            halts_norm = halts / (self.queue_norm if self.queue_norm > 0 else 1.0)
            queue_norm = queue_avg / (self.queue_norm if self.queue_norm > 0 else 1.0)

            reward = - (REWARD_WEIGHTS.get("halts", 1.0) * halts_norm + REWARD_WEIGHTS.get("queue", 0.0) * queue_norm)

            if USE_WAITING_TIME_IN_REWARD:
                waiting = sum(traci.lane.getWaitingTime(l) for l in self.controlled_lanes)
                denom = max(1.0, len(self.controlled_lanes) * (EPISODE_LENGTH if EPISODE_LENGTH>0 else 1.0))
                waiting_norm = waiting / denom
                reward -= REWARD_WEIGHTS.get("waiting", 0.0) * waiting_norm

            if self.elapsed_phase_time < self.min_green:
                reward -= REWARD_WEIGHTS.get("switch_penalty", SWITCH_PENALTY)

            return float(reward)
        except Exception as e:
            print(f"Error in _compute_reward: {e}")
            return 0.0

    def _can_switch_to(self, action_index:int) -> bool:
        try:
            if not self.safe_switch:
                return True
            if not self.phase_defs:
                return False

            cur_abs = self.allowed_phases[self.current_phase_index]
            cand_abs = self.allowed_phases[action_index]
            # protect against out-of-range indices
            if cur_abs >= len(self.phase_defs) or cand_abs >= len(self.phase_defs):
                return True

            cur_state = self.phase_defs[cur_abs].state
            cand_state = self.phase_defs[cand_abs].state

            if len(cur_state) < len(self.controlled_lanes) or len(cand_state) < len(self.controlled_lanes):
                return True

            for i, lane in enumerate(self.controlled_lanes):
                cur_c = cur_state[i]
                cand_c = cand_state[i]
                if (cur_c in ('G','g')) and (cand_c in ('r','R')):
                    try:
                        vehs = traci.lane.getLastStepVehicleIDs(lane)
                        lane_len = self.lane_lengths.get(lane, 999.0)
                        for v in vehs:
                            try:
                                pos = traci.vehicle.getLanePosition(v)
                            except Exception:
                                continue
                            dist_to_end = lane_len - pos
                            if dist_to_end <= self.safe_distance:
                                return False
                    except Exception:
                        return False
            return True
        except Exception:
            return False

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, dict]:
        done = False
        info = {}
        try:
            self.step_count += 1
            self.elapsed_phase_time += 1

            switch_happened = False
            if (action_index != self.current_phase_index) and (self.elapsed_phase_time >= self.min_green):
                if self._can_switch_to(action_index):
                    self.current_phase_index = int(action_index)
                    traci.trafficlight.setPhase(self.tls_id, int(self.allowed_phases[self.current_phase_index]))
                    self.elapsed_phase_time = 0
                    switch_happened = True

            traci.simulationStep()

            state = self._get_state()
            reward = self._compute_reward()

            if (self.step_count >= self.max_steps) or (traci.simulation.getMinExpectedNumber() == 0):
                done = True

            vehs = traci.vehicle.getIDList()
            halts = sum(int(traci.lane.getLastStepHaltingNumber(l)) for l in self.controlled_lanes)
            avg_speed = np.mean([max(0.0, traci.lane.getLastStepMeanSpeed(l)) for l in self.controlled_lanes]) if self.controlled_lanes else 0.0
            total_waiting = sum(traci.lane.getWaitingTime(l) for l in self.controlled_lanes)
            avg_queue = np.mean([traci.lane.getLastStepHaltingNumber(l) for l in self.controlled_lanes]) if self.controlled_lanes else 0.0
            num_stops = 0
            for v in vehs:
                try:
                    sp = traci.vehicle.getSpeed(v)
                    if sp < 0.01:
                        num_stops += 1
                except Exception:
                    continue

            info = {
                "vehicles": len(vehs),
                "halts": int(halts),
                "avg_speed": float(avg_speed),
                "total_waiting": float(total_waiting),
                "avg_queue": float(avg_queue),
                "num_stops": int(num_stops),
                "current_phase": int(self.allowed_phases[self.current_phase_index]),
                "elapsed_phase_time": int(self.elapsed_phase_time),
                "minExpected": int(traci.simulation.getMinExpectedNumber()),
                "switch": switch_happened
            }
        except TraCIException as e:
            print(f"TraCIException in env.step: {e}")
            try:
                self._close_sumo()
            except:
                pass
            return np.zeros(self.state_dim if self.state_dim else 1, dtype=np.float32), 0.0, True, {"error": str(e)}
        except Exception as e:
            print(f"Error in step: {e}")
            return np.zeros(self.state_dim if self.state_dim else 1, dtype=np.float32), 0.0, True, {"error": str(e)}

        return state, reward, done, info

    def close(self):
        self._close_sumo()


Transition = namedtuple('Transition', ('state','action','reward','next_state','done'))

class ReplayBuffer:
    def __init__(self, capacity=BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

def build_mlp(input_dim:int, output_dim:int, hidden:int=256, n_layers:int=2):
    layers=[]
    dims=[input_dim]+[hidden]*n_layers+[output_dim]
    for i in range(len(dims)-2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)

class DQNAgent:
    def __init__(self, state_dim:int, action_dim:int, device:torch.device):
        self.device = device
        self.state_dim = state_dim
        self.action_dim_val = action_dim

        self.online = build_mlp(state_dim, action_dim).to(device)
        self.target = build_mlp(state_dim, action_dim).to(device)
        self.target.load_state_dict(self.online.state_dict())

        self.opt = optim.Adam(self.online.parameters(), lr=LR)
        self.replay = ReplayBuffer(BUFFER_CAPACITY)
        self.loss_fn = nn.SmoothL1Loss()

    @property
    def action_dim(self):
        return self.action_dim_val

    def select_action(self, state:np.ndarray, eps:float) -> int:
        if random.random() < eps:
            return random.randrange(0, self.action_dim_val)
        st = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online(st)
            return int(torch.argmax(q, dim=1).item())

    def push(self, s, a, r, s2, done):
        self.replay.push(s, a, r, s2, done)

    def update(self, batch_size):
        if len(self.replay) < batch_size:
            return None

        trans = self.replay.sample(batch_size)
        s = torch.tensor(np.array(trans.state), dtype=torch.float32, device=self.device)
        a = torch.tensor(trans.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(trans.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(np.array(trans.next_state), dtype=torch.float32, device=self.device)
        d = torch.tensor(trans.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_vals = self.online(s).gather(1, a)
        with torch.no_grad():
            online_next = self.online(s2)
            next_actions = torch.argmax(online_next, dim=1, keepdim=True)
            q_next_target = self.target(s2).gather(1, next_actions)
            q_target = r + (1.0 - d) * GAMMA * q_next_target

        loss = self.loss_fn(q_vals, q_target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        return float(loss.item())

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path, total_steps: int = 0):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            'online': self.online.state_dict(),
            'target': self.target.state_dict(),
            'opt': self.opt.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim_val,
            'total_steps': total_steps
        }, path)

    def load(self, path) -> int:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.online.load_state_dict(checkpoint['online'])
        self.target.load_state_dict(checkpoint['target'])
        self.opt.load_state_dict(checkpoint['opt'])
        total_steps = checkpoint.get('total_steps', 0)
        print(f"Model loaded from {path}, resuming from step {total_steps}")
        return total_steps

def epsilon_by_step(step:int) -> float:
    return float(EPS_END + (EPS_START - EPS_END) * np.exp(-1. * step / EPS_DECAY))

def evaluate_policy(sumo_cfg:str, agent:DQNAgent, episodes:int=EVAL_EPISODES, episode_len:int=EPISODE_LENGTH, baseline:Optional[str]=None):
    env = TrafficEnv(sumo_cfg, use_gui=False, max_steps=episode_len, use_speed=USE_SPEED_IN_STATE)
    returns = []
    halts_list = []
    speeds = []
    waiting_times = []
    delays = []
    avg_queues = []
    num_stops_list = []
    try:
        for ep in range(episodes):
            s = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            total_halts = 0
            total_speed = 0.0
            total_waiting = 0.0
            total_delay = 0.0
            total_avg_queue = 0.0
            total_num_stops = 0

            if baseline == 'fixed':
                phase_idx = 0
                phase_timer = 0
                duration = BASELINE_PHASE_DURATION

            while not done and steps < episode_len:
                if baseline == 'fixed':
                    action = phase_idx
                    phase_timer += 1
                    if phase_timer >= duration:
                        phase_idx = (phase_idx + 1) % env.action_dim
                        phase_timer = 0
                elif baseline == 'random':
                    action = random.randint(0, env.action_dim - 1)
                else:
                    action = agent.select_action(s, eps=0.0)

                s, r, done, info = env.step(action)
                total_reward += r
                total_halts += info.get('halts', 0)
                total_speed += info.get('avg_speed', 0.0)
                total_waiting += info.get('total_waiting', 0.0)
                total_delay += info.get('total_waiting', 0.0)
                total_avg_queue += info.get('avg_queue', 0.0)
                total_num_stops += info.get('num_stops', 0)
                steps += 1

            returns.append(total_reward)
            halts_list.append(total_halts / max(1, steps))
            speeds.append(total_speed / max(1, steps))
            waiting_times.append(total_waiting / max(1, steps))
            delays.append(total_delay)
            avg_queues.append(total_avg_queue / max(1, steps))
            num_stops_list.append(total_num_stops / max(1, steps))
    except Exception as e:
        print(f"Error during evaluation: {e}")
    finally:
        env.close()

    metrics = {
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_halts": float(np.mean(halts_list)) if halts_list else 0.0,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "mean_waiting": float(np.mean(waiting_times)) if waiting_times else 0.0,
        "mean_delay": float(np.mean(delays)) if delays else 0.0,
        "mean_queue": float(np.mean(avg_queues)) if avg_queues else 0.0,
        "mean_stops": float(np.mean(num_stops_list)) if num_stops_list else 0.0
    }
    return metrics

def plot_metrics(csv_path: str, save_png: str):
    try:
        import pandas as pd
    except Exception:
        print("pandas required for plotting. Install pandas or skip plotting.")
        return

    if not os.path.exists(csv_path):
        print("CSV log not found for plotting:", csv_path)
        return

    df = pd.read_csv(csv_path)
    eval_df = df.dropna(subset=['eval_return', 'eval_waiting', 'eval_queue', 'eval_stops'], how='any').copy()
    if eval_df.empty:
        # fallback: try to find any eval_return present numerically
        if 'eval_return' in df.columns:
            eval_df = df[df['eval_return'].notna()].copy()
    if eval_df.empty:
        print("No evaluation data in CSV for plotting.")
        return

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    if 'env_steps' in eval_df.columns and 'eval_return' in eval_df.columns:
        axs[0].plot(eval_df['env_steps'], eval_df['eval_return'], marker='o', label='Agent Return')
        if 'baseline_fixed_return' in eval_df.columns:
            axs[0].plot(eval_df['env_steps'], eval_df['baseline_fixed_return'], marker='x', label='Fixed Return')
        if 'baseline_random_return' in eval_df.columns:
            axs[0].plot(eval_df['env_steps'], eval_df['baseline_random_return'], marker='^', label='Random Return')
        axs[0].set_ylabel('Return')
        axs[0].legend()

    if 'env_steps' in eval_df.columns and 'eval_waiting' in eval_df.columns:
        axs[1].plot(eval_df['env_steps'], eval_df['eval_waiting'], marker='o', label='Agent Waiting')
        if 'baseline_fixed_waiting' in eval_df.columns:
            axs[1].plot(eval_df['env_steps'], eval_df['baseline_fixed_waiting'], marker='x', label='Fixed Waiting')
        if 'baseline_random_waiting' in eval_df.columns:
            axs[1].plot(eval_df['env_steps'], eval_df['baseline_random_waiting'], marker='^', label='Random Waiting')
        axs[1].set_ylabel('Avg Waiting Time')
        axs[1].legend()

    if 'env_steps' in eval_df.columns and 'eval_queue' in eval_df.columns:
        axs[2].plot(eval_df['env_steps'], eval_df['eval_queue'], marker='o', label='Agent Queue')
        if 'baseline_fixed_queue' in eval_df.columns:
            axs[2].plot(eval_df['env_steps'], eval_df['baseline_fixed_queue'], marker='x', label='Fixed Queue')
        if 'baseline_random_queue' in eval_df.columns:
            axs[2].plot(eval_df['env_steps'], eval_df['baseline_random_queue'], marker='^', label='Random Queue')
        axs[2].set_ylabel('Avg Queue Length')
        axs[2].legend()

    if 'env_steps' in eval_df.columns and 'eval_stops' in eval_df.columns:
        axs[3].plot(eval_df['env_steps'], eval_df['eval_stops'], marker='o', label='Agent Stops')
        if 'baseline_fixed_stops' in eval_df.columns:
            axs[3].plot(eval_df['env_steps'], eval_df['baseline_fixed_stops'], marker='x', label='Fixed Stops')
        if 'baseline_random_stops' in eval_df.columns:
            axs[3].plot(eval_df['env_steps'], eval_df['baseline_random_stops'], marker='^', label='Random Stops')
        axs[3].set_ylabel('Avg Stops')
        axs[3].set_xlabel('Env Steps')
        axs[3].legend()

    plt.tight_layout()
    plt.savefig(save_png)
    plt.close()
    print("Saved plot to:", save_png)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    save_config(CFG, os.path.join(CHECKPOINT_DIR, "config_phase4.json"))

    env_tmp = TrafficEnv(SUMO_CONFIG, use_gui=USE_GUI, max_steps=EPISODE_LENGTH, use_speed=USE_SPEED_IN_STATE,
                         safe_switch=USE_SAFE_SWITCH, safe_distance=SAFE_SWITCH_DISTANCE)
    s0 = env_tmp.reset(verbose=False)
    state_dim = env_tmp.state_dim
    action_dim = env_tmp.action_dim
    controlled_lanes = len(env_tmp.controlled_lanes)
    print(f"State dim: {state_dim} | Action dim: {action_dim} | Controlled lanes: {controlled_lanes}")
    env_tmp.close()

    agent = DQNAgent(state_dim, action_dim, device)

    total_steps = 0
    if RESUME_TRAINING and RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        try:
            total_steps = agent.load(RESUME_CHECKPOINT)
            print("Resumed from:", RESUME_CHECKPOINT)
        except Exception as e:
            print("Resume load failed:", e)

    csv_file = open(LOG_CSV, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow([
        'env_steps', 'episode', 'episode_reward', 'episode_len',
        'eval_return', 'eval_halts', 'eval_speed', 'eval_waiting', 'eval_delay', 'eval_queue', 'eval_stops',
        'baseline_fixed_return', 'baseline_fixed_halts', 'baseline_fixed_speed', 'baseline_fixed_waiting', 'baseline_fixed_delay', 'baseline_fixed_queue', 'baseline_fixed_stops',
        'baseline_random_return', 'baseline_random_halts', 'baseline_random_speed', 'baseline_random_waiting', 'baseline_random_delay', 'baseline_random_queue', 'baseline_random_stops',
        'epsilon', 'loss'
    ])

    pbar = trange(total_steps, MAX_TRAIN_STEPS, desc="Env Steps")
    env = TrafficEnv(SUMO_CONFIG, use_gui=USE_GUI, max_steps=EPISODE_LENGTH, use_speed=USE_SPEED_IN_STATE,
                     safe_switch=USE_SAFE_SWITCH, safe_distance=SAFE_SWITCH_DISTANCE)

    episode = 0
    gradient_steps = 0
    best_eval = -1e9

    try:
        while total_steps < MAX_TRAIN_STEPS:
            episode += 1
            state = env.reset()
            done = False
            ep_reward = 0.0
            ep_len = 0
            ep_loss = 0.0
            loss_count = 0

            while (not done) and (ep_len < env.max_steps) and (total_steps < MAX_TRAIN_STEPS):
                eps = epsilon_by_step(total_steps)

                if state.shape[0] != state_dim:
                    st = np.zeros(state_dim, dtype=np.float32)
                    st[:min(state_dim, state.shape[0])] = state[:min(state_dim, state.shape[0])]
                    state = st

                action = agent.select_action(state, eps)

                try:
                    next_state, reward, done, info = env.step(action)
                except TraCIException as e:
                    print("TraCIException during step:", e)
                    try:
                        env.close()
                    except:
                        pass
                    time.sleep(0.2)
                    env = TrafficEnv(SUMO_CONFIG, use_gui=USE_GUI, max_steps=EPISODE_LENGTH, use_speed=USE_SPEED_IN_STATE,
                                     safe_switch=USE_SAFE_SWITCH, safe_distance=SAFE_SWITCH_DISTANCE)
                    state = env.reset()
                    continue

                agent.push(state, action, reward, next_state, float(done))
                state = next_state
                ep_reward += reward
                ep_len += 1
                total_steps += 1
                pbar.update(1)
                pbar.set_postfix({'eps': f'{eps:.3f}', 'reward': f'{ep_reward:.1f}', 'step': total_steps})

                if total_steps > TRAIN_START and len(agent.replay) >= BATCH_SIZE:
                    loss = agent.update(BATCH_SIZE)
                    if loss is not None:
                        ep_loss += loss
                        loss_count += 1
                    gradient_steps += 1
                    if gradient_steps % TARGET_UPDATE_FREQ == 0:
                        agent.sync_target()

                if total_steps % EVAL_EVERY == 0 and total_steps > 0:
                    print("\n" + ("="*40))
                    print(f"[Evaluation] env_steps={total_steps}, epsilon={eps:.3f}")
                    print("="*40)
                    try:
                        env.close()
                    except:
                        pass
                    time.sleep(0.12)

                    agent_metrics = evaluate_policy(SUMO_CONFIG, agent, episodes=EVAL_EPISODES, episode_len=EPISODE_LENGTH, baseline=None)
                    fixed_metrics = evaluate_policy(SUMO_CONFIG, agent, episodes=EVAL_EPISODES, episode_len=EPISODE_LENGTH, baseline='fixed')
                    random_metrics = evaluate_policy(SUMO_CONFIG, agent, episodes=EVAL_EPISODES, episode_len=EPISODE_LENGTH, baseline='random')
                    avg_loss = ep_loss / loss_count if loss_count > 0 else 0.0

                    writer.writerow([
                        total_steps, episode, ep_reward, ep_len,
                        agent_metrics['mean_return'], agent_metrics['mean_halts'], agent_metrics['mean_speed'], agent_metrics['mean_waiting'], agent_metrics['mean_delay'], agent_metrics['mean_queue'], agent_metrics['mean_stops'],
                        fixed_metrics['mean_return'], fixed_metrics['mean_halts'], fixed_metrics['mean_speed'], fixed_metrics['mean_waiting'], fixed_metrics['mean_delay'], fixed_metrics['mean_queue'], fixed_metrics['mean_stops'],
                        random_metrics['mean_return'], random_metrics['mean_halts'], random_metrics['mean_speed'], random_metrics['mean_waiting'], random_metrics['mean_delay'], random_metrics['mean_queue'], random_metrics['mean_stops'],
                        eps, avg_loss
                    ])
                    csv_file.flush()

                    print(f"[Eval Agent] return={agent_metrics['mean_return']:.2f} | halts={agent_metrics['mean_halts']:.2f} | speed={agent_metrics['mean_speed']:.2f} | waiting={agent_metrics['mean_waiting']:.2f}")
                    print(f"[Eval Fixed] return={fixed_metrics['mean_return']:.2f} | waiting={fixed_metrics['mean_waiting']:.2f}")
                    print(f"[Eval Random] return={random_metrics['mean_return']:.2f} | waiting={random_metrics['mean_waiting']:.2f}")

                    if agent_metrics['mean_return'] > best_eval:
                        best_eval = agent_metrics['mean_return']
                        ckpt_path = os.path.join(CHECKPOINT_DIR, f"dqn_best_{total_steps}.pth")
                        agent.save(ckpt_path, total_steps)
                        print(f"Saved best checkpoint: {ckpt_path}")

                    latest_path = os.path.join(CHECKPOINT_DIR, "dqn_latest.pth")
                    agent.save(latest_path, total_steps)

                    env = TrafficEnv(SUMO_CONFIG, use_gui=USE_GUI, max_steps=EPISODE_LENGTH, use_speed=USE_SPEED_IN_STATE,
                                     safe_switch=USE_SAFE_SWITCH, safe_distance=SAFE_SWITCH_DISTANCE)
                    state = env.reset()

            avg_loss_ep = ep_loss / loss_count if loss_count > 0 else None
            print("\n" + ("-"*30))
            print(f"Episode {episode} | env_steps={total_steps} | ep_reward={ep_reward:.2f} | ep_len={ep_len} | avg_loss={avg_loss_ep}")
            print("-"*30 + "\n")
            writer.writerow([total_steps, episode, ep_reward, ep_len] + [None] * 21 + [epsilon_by_step(total_steps), avg_loss_ep])
            csv_file.flush()

        agent.save(os.path.join(CHECKPOINT_DIR, "dqn_final.pth"), total_steps)
        print("Training finished. Final model saved.")

    except KeyboardInterrupt:
        print("Interrupted by user. Saving model...")
        agent.save(os.path.join(CHECKPOINT_DIR, "dqn_interrupted.pth"), total_steps)
    except Exception as e:
        print("Training error:", e)
        try:
            agent.save(os.path.join(CHECKPOINT_DIR, "dqn_error.pth"), total_steps)
        except Exception:
            pass
    finally:
        try:
            env.close()
        except:
            pass
        csv_file.close()
        pbar.close()

    plot_metrics(LOG_CSV, os.path.join(PLOT_DIR, "training_plot.png"))

if __name__ == "__main__":
    train()
