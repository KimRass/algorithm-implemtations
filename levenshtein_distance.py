def weighted_levenshtein_distance(ref, hypo, sub_cost=1, del_cost=1, ins_cost=1):
    """
    Compute the Levenshtein distance between two strings.
    """
    ref_len = len(ref)
    hyp_len = len(hypo)

    # Create a matrix.
    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    # dp[i][j] represents the minimum cost of transforming the first `𝑖` characters of the reference string into the first `j` characters of the hypothesis string.
    # Initialize the matrix.
    for i in range(ref_len + 1):
        dp[i][0] = i * del_cost
    # Column 0: Transforming the first 𝑖 i characters of the reference into an empty string requires 𝑖 × Deletion Cost i×Deletion Cost.
    for j in range(hyp_len + 1):
        dp[0][j] = j * ins_cost
    # Row 0: Transforming the empty string into the first 𝑗 j characters of the hypothesis requires 𝑗 × Insertion Cost j×Insertion Cost.

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if ref[i - 1] == hypo[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No cost for a match.
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1] + sub_cost,
                    dp[i - 1][j] + del_cost,
                    dp[i][j - 1] + ins_cost,
                )
    pprint(dp)
    return dp[ref_len][hyp_len]


if __name__ == "__main__":
    ref = "levnte"
    hypo = "net"
    weighted_levenshtein_distance(ref, hypo)
    # weighted_levenshtein_distance(ref, hypo, 1, 2, 3)
    from pprint import pprint
