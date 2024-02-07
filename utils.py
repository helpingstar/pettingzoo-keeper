import torch.nn as nn


def schedule_matches(n: int = 4, is_symmetry: bool = False):
    # This function generates the schedule of matches for n teams.
    assert n % 2 == 0, "n must be even"

    # List to store all rounds
    schedule = []

    for round in range(n - 1):
        round_matches = []
        for match in range(n // 2):
            team1 = (round + match) % (n - 1)
            team2 = (n - 1 - match + round) % (n - 1)
            if (
                match == 0
            ):  # Always include the last team in the first match of each round
                team2 = n - 1
            # Adjusting team number to be 1-indexed instead of 0-indexed
            round_matches.append((min(team1, team2), max(team1, team2)))
        schedule.append(round_matches)

    if not is_symmetry:
        flipped_match = []
        for round in schedule:
            round_matches = []
            for team1, team2 in round:
                round_matches.append((team2, team1))
            flipped_match.append(round_matches)
        schedule.extend(flipped_match)
    return schedule


def copy_network(src: nn.Module, dest: nn.Module):
    dest.load_state_dict(src.state_dict())


if __name__ == "__main__":
    print(schedule_matches())
