from collections import Counter
from difflib import SequenceMatcher


def similar(a: str, b: str, threshold: float = 0.6) -> bool:
    """Checks if two strings are similar based on a threshold."""
    return SequenceMatcher(None, a, b).ratio() >= threshold


def group_similar_plates(plates: list[str], threshold: float = 0.6) -> list[list[str]]:
    """Groups similar plates based on a similarity threshold."""
    groups = []
    for plate in plates:
        if plate == "Unable to read" or len(plate) < 3:
            continue
        added = False
        for g in groups:
            if similar(plate, g[0], threshold):
                g.append(plate)
                added = True
                break
        if not added:
            groups.append([plate])
    return groups


def get_unique_plates(plates: list[str]) -> list[str]:
    """Extracts unique plate numbers from a list of plates."""
    groups = group_similar_plates(plates)
    unique_plates = []
    for g in groups:
        freq = Counter(g)
        best = max(freq, key=lambda x: (freq[x], len(x)))
        unique_plates.append(best)
    return unique_plates


def get_plate_number(label: str) -> str:
    """Extracts the plate number from the label."""
    return label.split(":")[-1].strip()
