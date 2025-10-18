from typing import List, Dict, Sequence

PHONEME_DICT = {
    "ABACUS": ["A", "B", "AH", "K", "S"],
    "BOOK": ["B", "UH", "K"],
    "TH": ["T", "H"],
    "DER": ["D", "EH", "R"],
    "THE": ["DH", "EH"],
    "THEIR": ["DH", "EH", "R"],
    "THERE": ["DH", "EH", "R"],
    "TOMATO": ["T", "AH", "M", "EY", "T", "OW"],
    "TAMA": ["T", "AH", "M", "AH"],
    "TOM": ["T", "AH", "M"]
}

def build_length_map(word_dict: Dict[str, List[str]]) -> Dict[int, List[tuple]]:
    """
    Builds a helper structure:
      length_map[L] = list of (word, phoneme_sequence)
    where L = len(phoneme_sequence).
    """
    length_map: Dict[int, List[tuple]] = {}
    for word, ph_list in word_dict.items():
        length_map.setdefault(len(ph_list), []).append((word, ph_list))
    return length_map

def find_word_combos_with_pronunciation(phonemes: Sequence[str]) -> List[List[str]]:
    """
    Given a sequence of phonemes, returns ALL possible combinations of words
    from PHONEME_DICT that produce exactly this phoneme sequence (in order).
    
    Example:
      If phonemes = ["DH", "EH", "R"], valid outputs could be:
         [["THEIR"], ["THERE"]]
      (assuming 'THEIR' and 'THERE' share the same phoneme sequence).
    """
    # Create the helper structure
    length_map = build_length_map(PHONEME_DICT)
    
    # DFS with memoization to find all solutions
    memo = {}  # key = start_index in 'phonemes', value = list of valid combos
    
    def backtrack(i: int) -> List[List[str]]:
        # If we’ve reached the end of the phoneme list, one valid combination is "no words left to match"
        if i == len(phonemes):
            return [[]]  # One complete (empty) solution from here

        if i in memo:
            return memo[i]
        
        results = []
        
        # Try all possible segment lengths L that won't go out of bounds
        for L, word_pairs in length_map.items():
            end = i + L
            if end <= len(phonemes):
                segment = phonemes[i:end]
                # Compare with each (word, phoneme_list) of the same length
                for word, ph_list in word_pairs:
                    if ph_list == list(segment):
                        # If the segment matches this word’s phoneme sequence,
                        # recurse on the remainder
                        for suffix_combo in backtrack(end):
                            results.append([word] + suffix_combo)
        
        memo[i] = results
        return results

    return backtrack(0)

# Tests
def test_find_word_combos_with_pronunciation():
    test_cases = [
        # Single match: "THEIR" or "THERE" => both have same phonemes
        {
            "input": ["DH", "EH", "R"],
            "expected": [["THEIR"], ["THERE"]]
        },
        # Combination of two words: "TAMA" + "BOOK"
        {
            "input": ["T", "AH", "M", "AH", "B", "UH", "K"],
            "expected": [["TAMA", "BOOK"]]
        },
        # No possible combination
        {
            "input": ["A", "B", "UH", "K", "S", "XYZ"],
            "expected": []  # "XYZ" doesn't match any dictionary entry
        }
    ]
    
    for case in test_cases:
        inp = case["input"]
        expected = case["expected"]
        result = find_word_combos_with_pronunciation(inp)
        
        # Convert both 'result' and 'expected' to sets of tuples for easy comparison
        result_set = {tuple(r) for r in result}
        expected_set = {tuple(e) for e in expected}
        
        if result_set == expected_set:
            print(f"PASS: Input={inp} => {result}")
        else:
            print(f"FAIL: Input={inp}\n  Expected={expected}\n  Got={result}")

if __name__ == "__main__":
    test_find_word_combos_with_pronunciation()
