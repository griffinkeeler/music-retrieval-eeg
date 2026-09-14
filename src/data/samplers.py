from collections import defaultdict
import math
import random
from torch.utils.data import Sampler


class WithinSongBatchSampler(Sampler[list[int]]):
    def __init__(self, metadata, batch_size, drop_last=False, seed=0):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = random.Random(seed)

        self.song_to_indices = defaultdict(list)
        for idx, row in metadata.reset_index(drop=True).iterrows():
            self.song_to_indices[int(row["song_id"])].append(idx)

        self.song_indices = sorted(self.song_to_indices.keys())

    def __iter__(self):
        # Make a fresh set of indices for each song
        song_pools = {
            song_id: indices.copy()
            for song_id, indices in self.song_to_indices.items()
        }

        # Shuffle indices inside each song
        for indices in song_pools.values():
            self.rng.shuffle(indices)

        # Find songs with samples left
        available_songs = [s for s, idxs in song_pools.items() if len(idxs) > 0]

        # Continues until every song has run out of samples
        while available_songs:
            # Pick a random song
            song_id = self.rng.choice(available_songs)
            batch = []

            # Fill the batch from that song only
            while song_pools[song_id] and len(batch) < self.batch_size:
                batch.append(song_pools[song_id].pop())

            # Decide whether to yield the batch
            if len(batch) == self.batch_size or (len(batch) > 0 and not self.drop_last):
                yield batch

            # Refresh available songs
            available_songs = [s for s, idxs in song_pools.items() if len(idxs) > 0]

    def __len__(self):
        # Calculates the number of batches per song, then adds them up
        total = 0
        for indices in self.song_to_indices.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)
        return total


class AcrossSongBatchSampler(Sampler[list[int]]):
    def __init__(self, metadata, batch_size, drop_last=False, seed=0):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = random.Random(seed)

        self.song_to_indices = defaultdict(list)
        for idx, row in metadata.reset_index(drop=True).iterrows():
            self.song_to_indices[int(row["song_id"])].append(idx)

        self.song_ids = sorted(self.song_to_indices.keys())
        self.total_examples = sum(len(indices) for indices in self.song_to_indices.values())

    def __iter__(self):
        song_pools = {
            song_id: indices.copy()
            for song_id, indices in self.song_to_indices.items()
        }

        for indices in song_pools.values():
            self.rng.shuffle(indices)

        available_songs = [song_id for song_id, indices in song_pools.items() if indices]

        while available_songs:
            self.rng.shuffle(available_songs)
            batch = []

            # Pull from as many different songs as possible before reusing one.
            while len(batch) < self.batch_size and available_songs:
                next_available_songs = []

                for song_id in available_songs:
                    if len(batch) >= self.batch_size:
                        next_available_songs.append(song_id)
                        continue

                    if song_pools[song_id]:
                        batch.append(song_pools[song_id].pop())

                    if song_pools[song_id]:
                        next_available_songs.append(song_id)

                available_songs = next_available_songs

                if available_songs and len(batch) < self.batch_size:
                    self.rng.shuffle(available_songs)

            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch

            available_songs = [song_id for song_id, indices in song_pools.items() if indices]

    def __len__(self):
        if self.drop_last:
            return self.total_examples // self.batch_size
        return math.ceil(self.total_examples / self.batch_size)
