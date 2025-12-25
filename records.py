import json
import os
import time

class RecordManager:
    """Handles saving and loading of game and tournament records."""
    def __init__(self, records_file="game_history.json"):
        self.filename = records_file
        self.history = self.load_history()

    def load_history(self):
        if not os.path.exists(self.filename):
            return {"games": [], "tournaments": []}
        try:
            with open(self.filename, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"games": [], "tournaments": []}

    def save_history(self):
        try:
            with open(self.filename, "w") as f:
                json.dump(self.history, f, indent=4)
        except IOError as e:
            print(f"Error saving history: {e}")

    def add_game_record(self, record):
        """
        record: dict containing:
          - timestamp
          - p1_name, p1_level
          - p2_name, p2_level
          - m, n, k
          - winner (0, 1, 2)
          - win_reason
          - moves_count
          - mode (Quick/Tournament)
        """
        record['timestamp'] = record.get('timestamp', time.time())
        self.history["games"].append(record)
        self.save_history()

    def add_tournament_record(self, record):
        """
        record: dict containing:
          - timestamp
          - type (RoundRobin/Knockout)
          - players (list of names/levels)
          - winner_name
          - rounds/matches details
        """
        record['timestamp'] = record.get('timestamp', time.time())
        self.history["tournaments"].append(record)
        self.save_history()
