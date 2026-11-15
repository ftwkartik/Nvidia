'''
Algorithm
1. Start.
2. Read the values of n and k.
3. Create a DP table of size (n+1) × (k+1) and initialize it with zeros.
4. Fill the table using the following conditions:
   - If j = 0 or j = i, set C[i][j] = 1.
   - Otherwise, calculate:

C(i,j)=C(i-1,j-1)+C(i-1,j)


5. Repeat the process until all required values are calculated.
6. Display the binomial coefficient C(n,k).
7. Stop.   '''


def binomial_coefficient(n, k):

    # Create DP table
    dp = [[0 for j in range(k + 1)]
          for i in range(n + 1)]

    # Fill the DP table
    for i in range(n + 1):
        for j in range(min(i, k) + 1):

            # Base condition
            if j == 0 or j == i:
                dp[i][j] = 1

            else:
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]

    return dp[n][k]


# Main program
n = int(input("Enter value of n: "))
k = int(input("Enter value of k: "))

if n < 0 or k < 0 or k > n:
    print("Invalid input")
else:
    result = binomial_coefficient(n, k)
    print("Binomial Coefficient:", result)


'''
Complexity Analysis
- Time Complexity: (O(n x k))
- Space Complexity: (O(n x k))   '''