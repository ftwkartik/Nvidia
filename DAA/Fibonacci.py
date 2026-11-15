'''
Algorithm
1. Start.
2. Read the number of terms n.
3. Initialize a = 0, b = 1 and count = 0.
4. Repeat the following steps until n terms are generated:
   - Print a.
   - Calculate c = a + b.
   - Update a = b and b = c.
   - Increment count.
5. Display the total number of computational steps.
6. Stop.
'''


n = int(input("Enter the number of terms: "))

a, b = 0, 1
count = 0

print("Fibonacci Series:")

for i in range(n):
    print(a, end=" ")
    c = a + b
    a = b
    b = c
    count += 1

print("\nTotal computational steps:", count)


'''
Time Complexity: (O(n))

Space Complexity: (O(1))
'''