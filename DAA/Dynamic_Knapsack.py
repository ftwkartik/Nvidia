'''
Algorithm
1. Start.
2. Read the number of items and knapsack capacity.
3. Read the weight and profit of each item.
4. Create a DP table of size (n+1) × (W+1) and initialize it with zeros.
5. For each item and capacity:
   - If the item's weight is less than or equal to the current capacity, choose the maximum of:
     - Including the item.
     - Excluding the item.
   - Otherwise, exclude the item.
6. Store the maximum profit in the DP table.
7. Display the maximum profit.
8. Stop.  '''


def knapsack(W, weights, profits, n):

    # Create DP table
    dp = [[0 for _ in range(W + 1)]
          for _ in range(n + 1)]

    # Build DP table
    for i in range(1, n + 1):
        for w in range(W + 1):

            if weights[i - 1] <= w:
                dp[i][w] = max(
                    profits[i - 1] + dp[i - 1][w - weights[i - 1]],
                    dp[i - 1][w]
                )
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][W]


# Main program
n = int(input("Enter number of items: "))

weights = []
profits = []

for i in range(n):
    weights.append(int(input("Enter weight: ")))
    profits.append(int(input("Enter profit: ")))

W = int(input("Enter knapsack capacity: "))

max_profit = knapsack(W, weights, profits, n)

print("Maximum Profit:", max_profit)


'''
Complexity Analysis
- Time Complexity: (O(n x W))
- Space Complexity: (O(n x W))
Where:
- n = Number of items.
- W = Knapsack capacity       '''