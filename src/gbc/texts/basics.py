# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.


def plural_to_singular(string):
    # So that we do not need to install it unless needed
    from hbutils.string.plural import singular_form

    quantifiers = [
        "a",
        "an",
        "some",
        "many",
        "several",
        "few",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
    ]
    if " " in string:
        string_parts = string.split()
        if string_parts[0].lower() in quantifiers:
            string = " ".join(string_parts[1:])
    return singular_form(string).lower()


def remove_repeated_suffix(s: str) -> str:
    """
    Removes the repeated suffix from the string efficiently using Rolling Hash.
    """
    if not s:
        return s

    n = len(s)
    base = 257  # A prime number base for hashing
    mod = 10**9 + 7  # A large prime modulus to prevent overflow

    # Precompute prefix hashes and powers of the base
    prefix_hash = [0] * (n + 1)
    power = [1] * (n + 1)

    for i in range(n):
        prefix_hash[i + 1] = (prefix_hash[i] * base + ord(s[i])) % mod
        power[i + 1] = (power[i] * base) % mod

    def get_hash(left, right):
        return (prefix_hash[right] - prefix_hash[left] * power[right - left]) % mod

    max_k = 0  # To store the maximum k where suffix is repeated

    # Iterate over possible suffix lengths from 1 to n//2
    for k in range(1, n // 2 + 1):
        # Compare the last k characters with the k characters before them
        if get_hash(n - 2 * k, n - k) == get_hash(n - k, n):
            max_k = k  # Update max_k if a repeated suffix is found

    if max_k > 0:
        # Remove the extra occurrences of the suffix
        # Calculate how many times the suffix is repeated consecutively
        m = 2
        while max_k * (m + 1) <= n and get_hash(
            n - (m + 1) * max_k, n - m * max_k
        ) == get_hash(n - m * max_k, n - (m - 1) * max_k):
            m += 1
        # Remove (m-1) copies of the suffix
        s = s[: n - (m - 1) * max_k]

    return s
