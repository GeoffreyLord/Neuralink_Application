def generate_symmetric_sequence():
    import itertools

    digits = '0123456789'
    pairs = set(itertools.combinations_with_replacement(digits, 2))
    pairs = {''.join(pair) for pair in pairs}
    pairs |= {pair[::-1] for pair in pairs}  # Ensure all reverses are included

    sequence = '0'
    used_pairs = set()

    def can_add_pair(pair):
        return pair not in used_pairs and pair[::-1] not in used_pairs

    for pair in pairs:
        if can_add_pair(pair):
            sequence += pair[1]  # Append the second digit of the pair
            used_pairs.add(pair)

    # Ensure the sequence starts with '0' to include '00' correctly
    sequence = '0' + sequence[1:]
    return sequence

symmetric_sequence = generate_symmetric_sequence()
print(f"Symmetric Sequence: {symmetric_sequence}")
print(f"Length of the sequence: {len(symmetric_sequence)}")