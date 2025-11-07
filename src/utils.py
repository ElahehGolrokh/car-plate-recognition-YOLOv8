from difflib import SequenceMatcher


def similar(a: str, b: str, threshold: float = 0.6) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= threshold

def group_similar_plates(plates, threshold=0.6):
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

def get_unique_plates(plates):
    from collections import Counter
    groups = group_similar_plates(plates)
    unique_plates = []
    for g in groups:
        freq = Counter(g)
        best = max(freq, key=lambda x: (freq[x], len(x)))
        unique_plates.append(best)
    return unique_plates


def get_plate_number(label: str) -> str:
    # Extract the plate number from the label
    return label.split(":")[-1].strip()
