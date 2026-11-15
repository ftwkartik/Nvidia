'''
Algorithm
1. Start.
2. Read the number of items and knapsack capacity.
3. For each item, read its weight and profit.
4. Calculate the profit-to-weight ratio for each item.

Ratio = {Profit}/{Weight}

5. Sort all items in descending order of their profit-to-weight ratio.
6. Initialize total profit to zero.
7. Select items according to the following conditions:
   - If the entire item fits, add it completely to the knapsack.
   - Otherwise, take the fraction of the item that fits.
8. Calculate the maximum total profit.
9. Display the total profit.
10. Stop.  '''


def fractional_knapsack(capacity, items):
    # Sort items by profit/weight ratio
    items.sort(key=lambda x: x[1] / x[0], reverse=True)

    total_profit = 0

    for weight, profit in items:

        if capacity == 0:
            break

        # Take the complete item
        if weight <= capacity:
            capacity -= weight
            total_profit += profit

        # Take the fraction of the item
        else:
            fraction = capacity / weight
            total_profit += profit * fraction
            capacity = 0

    return total_profit


# Main program
n = int(input("Enter number of items: "))
items = []

for i in range(n):
    weight = float(input("Enter weight: "))
    profit = float(input("Enter profit: "))
    items.append((weight, profit))

capacity = float(input("Enter knapsack capacity: "))

max_profit = fractional_knapsack(capacity, items)

print("Maximum Profit:", max_profit)


''' 
Complexity Analysis
- Time Complexity: (O(nlog n)), due to sorting the items based on their profit-to-weight ratio.
- Space Complexity: (O(n)), for storing the items. '''