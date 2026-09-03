from collections import deque
from typing import Callable, List
import numpy as np
import pandas as pd

class ReplayBuffer:
    def __init__(self, populateFunc: Callable, capacity: int = 256):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        populateFunc(self.buffer)
    
    def add(self, raw_obs, obs, action, reward, next_obs, done):
        self.buffer.append((raw_obs, obs, action, reward, next_obs, done))

    def sample_seq(self, seq_len: int, start_idx: int = 0):
        """
        Sample a sequence of experiences from the buffer.
        Args:
            seq_len: Length of the sequence to sample
            start_idx: Starting index within the buffer to sample from
        Returns:
            Tuple of (raw_obs_seq, obs_seq, action_seq, reward_seq, next_obs_seq, done_seq)
        """

        if len(self.buffer) < seq_len:
            raise ValueError("Not enough elements in buffer to sample the sequence.")
        if start_idx < 0 or start_idx + seq_len > len(self.buffer):
            raise IndexError("Start index out of bounds for sampling sequence.")
        seq = [self.buffer[i] for i in range(start_idx, start_idx + seq_len)]
        
        raw_obs_seq, obs_seq, action_seq, reward_seq, next_obs_seq, done_seq = zip(*seq)
        
        return (np.array(raw_obs_seq), np.array(obs_seq), np.array(action_seq), np.array(reward_seq),
                np.array(next_obs_seq), np.array(done_seq))

def parquet_populate(file_path: str, calculate_obs: Callable) -> Callable:
    def populate(buffer: deque):
        df = pd.read_parquet(file_path)
        for _, row in df.iterrows():
            buffer.append((row, calculate_obs(row),
                            np.ndarray(12, dtype=np.int64), 
                            row['reward'], 
                            row['next_obs'], 
                            row['done']
                            ))
    return populate
if __name__ == "__main__":
    # Example usage
    def populate(buffer):
        for i in range(300):
            buffer.append((i, i, i*2, i*3, i*4, False))
    
    replay_buffer = ReplayBuffer(populateFunc=populate, capacity=256)
    
    raw_obs_seq, obs_seq, action_seq, reward_seq, next_obs_seq, done_seq = replay_buffer.sample_seq(seq_len=10, start_idx=5)
    
    print("Sampled Raw Observation Sequence:", raw_obs_seq)
    print("Sampled Observation Sequence:", obs_seq)
    print("Sampled Action Sequence:", action_seq)
    print("Sampled Reward Sequence:", reward_seq)
    print("Sampled Next Observation Sequence:", next_obs_seq)
    print("Sampled Done Sequence:", done_seq)