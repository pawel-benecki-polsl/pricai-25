import os
from collections import defaultdict


def parse_filename(filename):
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    if len(parts) != 7:
        return None
    prefix = parts[0]
    index = parts[1]
    suffix = parts[2]
    if parts[3] != 'machine' or parts[5] != 'human':
        return None
    try:
        machine_judge = float(parts[4])
        human_judge = float(parts[6])
    except ValueError:
        return None
    return index, prefix, machine_judge, human_judge


def compute_statistics(directory):
    data = {}
    for filename in os.listdir(directory):
        if not filename.endswith('.png'):
            continue
        parsed = parse_filename(filename)
        if parsed is None:
            continue
        index, prefix, machine_judge, human_judge = parsed
        if index in data:
            # Verify consistency
            existing_prefix, existing_machine, existing_human = data[index]
            if prefix != existing_prefix or machine_judge != existing_machine or human_judge != existing_human:
                print(f"Inconsistent data for index {index} in file {filename}")
                continue
        else:
            data[index] = (prefix, machine_judge, human_judge)

    total_instances = len(data)
    if total_instances == 0:
        print("No valid data found.")
        return

    same_count = sum(1 for _, (prefix, _, _) in data.items() if prefix == 'same')
    different_count = total_instances - same_count

    print(f"Total unique instances: {total_instances}")
    print(f"Number of 'same': {same_count} ({same_count / total_instances * 100:.2f}%)")
    print(f"Number of 'different': {different_count} ({different_count / total_instances * 100:.2f}%)")

    # Distributions
    machine_dist = defaultdict(int)
    human_dist = defaultdict(int)
    for _, (_, mj, hj) in data.items():
        machine_dist[mj] += 1
        human_dist[hj] += 1

    print("\nMachine judge distribution:")
    for val, count in sorted(machine_dist.items()):
        print(f"  {val}: {count} ({count / total_instances * 100:.2f}%)")

    print("\nHuman judge distribution:")
    for val, count in sorted(human_dist.items()):
        print(f"  {val}: {count} ({count / total_instances * 100:.2f}%)")

    # For different
    different_cases = defaultdict(int)
    for _, (prefix, mj, hj) in data.items():
        if prefix == 'different':
            if mj == 0.0 and hj == 1.0:
                different_cases['machine_p0_human_p1'] += 1
            elif mj == 1.0 and hj == 0.0:
                different_cases['machine_p1_human_p0'] += 1
            elif mj == 0.0 and hj == 0.5:
                different_cases['machine_p0_human_tie'] += 1
            elif mj == 0.5 and hj == 0.0:
                different_cases['machine_tie_human_p0'] += 1
            elif mj == 1.0 and hj == 0.5:
                different_cases['machine_p1_human_tie'] += 1
            elif mj == 0.5 and hj == 1.0:
                different_cases['machine_tie_human_p1'] += 1
            elif mj == 0.5 and hj == 0.0:
                different_cases['machine_tie_human_p0'] += 1
            else:
                different_cases['other'] += 1

    print("\nIn 'different' cases:")
    for case, count in different_cases.items():
        print(f"  {case}: {count} ({count / different_count * 100:.2f}% of different)")


if __name__ == "__main__":
    directory = r"e:\experiments\2afc_comparison"
    compute_statistics(directory)