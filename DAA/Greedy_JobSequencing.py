'''
Algorithm
1. Start.
2. Read the number of jobs.
3. For each job, read its job ID, deadline and profit.
4. Sort all jobs in descending order of profit.
5. Find the maximum deadline.
6. Create an array of slots and initialize all slots as empty.
7. For each job:
   - Check the available slots from its deadline towards the first slot.
   - If an empty slot is found, assign the job to that slot.
   - Add its profit to the total profit.
8. Display the scheduled jobs and maximum profit.
9. Stop.
'''


def job_sequencing(jobs):
    # Sort jobs in descending order of profit
    jobs.sort(key=lambda x: x[2], reverse=True)

    max_deadline = max(job[1] for job in jobs)

    # Initialize time slots
    slots = [None] * max_deadline
    total_profit = 0

    # Schedule jobs
    for job in jobs:
        job_id, deadline, profit = job

        # Find an empty slot before deadline
        for j in range(min(deadline, max_deadline) - 1, -1, -1):
            if slots[j] is None:
                slots[j] = job_id
                total_profit += profit
                break

    print("Scheduled Jobs:", [j for j in slots if j is not None])
    print("Maximum Profit:", total_profit)


# Main program
n = int(input("Enter number of jobs: "))
jobs = []

for i in range(n):
    job_id = input("Enter Job ID: ")
    deadline = int(input("Enter Deadline: "))
    profit = int(input("Enter Profit: "))
    jobs.append((job_id, deadline, profit))

job_sequencing(jobs)



'''
Complexity Analysis
- Time Complexity: (O(nlog n + nD)), where (n) is the number of jobs and (D) is the maximum deadline.
- Space Complexity: (O(D)), for storing the available time slots.
'''