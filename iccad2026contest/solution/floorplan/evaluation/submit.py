"""Format solution for FloorSet contest submission."""
import json
import os
from typing import List, Tuple


def format_submission(solutions_path: str, output_path: str):
    """
    Convert solutions.json to FloorSet submission format.

    The submission format required by the contest evaluator:
    A list of placements (x, y, w, h) per test case.
    """
    with open(solutions_path) as f:
        solutions = json.load(f)

    submission = []
    for idx in sorted(solutions.keys(), key=lambda x: int(x)):
        placements = solutions[idx]['placements']
        submission.append({
            'test_id': int(idx),
            'placements': placements,
        })

    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)

    print(f"Submission saved to {output_path} ({len(submission)} test cases)")
    return submission
