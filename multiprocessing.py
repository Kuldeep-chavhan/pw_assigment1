#!/usr/bin/env python
# coding: utf-8

# Q1. What is multiprocessing in python? Why is it useful?

# What Is Multiprocessing?
# In the context of programming, particularly in Python, multiprocessing involves using multiple processes (not threads) to execute code concurrently.
# Unlike threads, which share the same memory space and are subject to the Global Interpreter Lock (GIL), processes run in separate memory spaces. This means they can truly parallelize tasks and utilize multiple CPU cores simultaneously.
# The GIL in Python prevents true parallel execution of threads, making multiprocessing an attractive alternative for CPU-bound and I/O-bound tasks.
# Why Is Multiprocessing Useful?
# Parallelism: By creating separate processes, you can take advantage of all available CPU cores. This is especially beneficial for computationally intensive tasks like numerical simulations, data processing, and machine learning.
# Avoiding GIL Limitations: The GIL restricts Python threads from executing Python code in parallel. Multiprocessing bypasses this limitation by running separate Python interpreter processes, each with its own memory space.
# Improved Performance: When you have tasks that can be parallelized, multiprocessing can significantly speed up execution. For example, if you’re analyzing large datasets or training machine learning models, multiprocessing can distribute the workload efficiently.
# Fault Isolation: Processes are isolated from each other. If one process encounters an error (e.g., a segmentation fault), it won’t affect other processes.
# Resource Utilization: Multiprocessing allows you to fully utilize available CPU resources, making your code more efficient.
# Example Usage:
# Let’s say you have a function that squares a number:

# Q2. What are the differences between multiprocessing and multithreading?

# Multiprocessing:
# Definition: Multiprocessing involves using two or more CPUs (central processing units) to increase the overall computing power of a system.
# How It Works: In a multiprocessing system, multiple processes run simultaneously. Each process has its own memory space and executes independently. These processes can communicate with each other through inter-process communication mechanisms.
# Advantages:
# Increases computing power by utilizing multiple processors.
# Suitable for tasks that require heavy computational power (e.g., scientific simulations, rendering, data processing).
# Disadvantages:
# Process creation can be time-consuming.
# Each process has its own address space, which can lead to higher memory usage.
# Multithreading:
# Definition: Multithreading focuses on generating multiple threads within a single process to increase computing speed.
# How It Works: In multithreading, a single process is divided into smaller threads, each of which can execute concurrently. These threads share the same memory space (address space) and can communicate directly with each other.
# Advantages:
# More efficient than multiprocessing for tasks within a single process.
# Threads share a common address space, which is memory-efficient.
# Disadvantages:
# Not classified into categories like multiprocessing.
# Thread creation is economical, but it can lead to synchronization issues (e.g., race conditions).
# Key Differences:
# Parallelism:
# Multiprocessing: Many processes run simultaneously.
# Multithreading: Many threads of a single process run simultaneously.
# Classification:
# Multiprocessing: Classified into symmetric and asymmetric multiprocessing.
# Multithreading: Not categorized into specific types.
# Process Creation:
# Multiprocessing: Time-consuming process creation.
# Multithreading: Economical process creation.
# Memory Space:
# Multiprocessing: Each process has its own separate address space.
# Multithreading: Threads share a common address space.
# Use Cases:
# Multiprocessing: Heavy computational tasks.
# Multithreading: Tasks within a single process (e.g., GUI responsiveness, web servers).

# Q3. Write a python code to create a process using the multiprocessing module.

# In[1]:


import multiprocessing

def worker_function(number):
    """A simple function that prints the square of a given number."""
    result = number * number
    print(f"Square of {number} is {result}")

if __name__ == "__main__":
    # Create a process that runs the worker_function with argument 5
    process = multiprocessing.Process(target=worker_function, args=(5,))
    
    # Start the process
    process.start()
    
    # Wait for the process to finish
    process.join()

    print("Main process continues executing...")


# Q4. What is a multiprocessing pool in python? Why is it used?

# A multiprocessing pool is a powerful construct provided by the multiprocessing module that allows you to efficiently manage a group of worker processes. These worker processes can execute tasks concurrently, taking advantage of multiple CPU cores. Here are the key points about multiprocessing pools:
# 
# What Is a Multiprocessing Pool?
# A multiprocessing pool is essentially a collection of worker processes that can be used to parallelize tasks.
# It provides an easy way to distribute work across multiple processes, especially when you have a large number of independent tasks to perform.
# How Does It Work?
# You create a pool of worker processes, specifying the desired number of processes.
# You submit tasks (functions) to the pool, and the pool automatically assigns them to available workers.
# The workers execute the tasks concurrently, potentially speeding up your computations significantly.
# Why Use Multiprocessing Pools?
# Parallelism: By using a pool, you can achieve parallel execution of tasks. Each worker in the pool operates independently, allowing you to process data faster.
# Resource Utilization: Pools manage the creation and destruction of worker processes, ensuring efficient use of system resources.
# Avoiding Global Interpreter Lock (GIL): Python’s Global Interpreter Lock prevents true parallel execution of threads. Multiprocessing pools bypass this limitation by creating separate processes, each with its own memory space and interpreter.
# Scalability: Pools are useful when you need to scale your computations across multiple cores or even multiple machines.
# Creating a Multiprocessing Pool in Python:
# You can create a pool using the multiprocessing.Pool class.
# Example:
# Python
# 
# import multiprocessing
# 
# def process_task(item):
#     # Your task logic here
#     return item * 2
# 
# if __name__ == "__main__":
#     data = [1, 2, 3, 4, 5]
#     with multiprocessing.Pool(processes=4) as pool:
#         results = pool.map(process_task, data)
#     print(results)
# AI-generated code. Review and use carefully. More info on FAQ.
# In this example, the process_task function is applied to each item in the data list using four worker processes.

# Q5. How can we create a pool of worker processes in python using the multiprocessing module?

# In[ ]:


import multiprocessing

def process_task(task_arg):
    # Your actual task logic goes here
    result = task_arg * 2
    return result

if __name__ == "__main__":
    num_processes = multiprocessing.cpu_count()  # Use all available cores
    with multiprocessing.Pool(processes=num_processes) as pool:
        input_data = [1, 2, 3, 4, 5]
        results = pool.map(process_task, input_data)
        print("Results:", results)


# Q6. Write a python program to create 4 processes, each process should print a different number using the
# multiprocessing module in python.

# In[ ]:


import multiprocessing

def print_number(number):
    print(f"Process {number}: My lucky number is {number}!")

if __name__ == "__main__":
    # Create four processes
    processes = []
    for i in range(1, 5):
        process = multiprocessing.Process(target=print_number, args=(i,))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    print("All processes have completed.")


# In[ ]:




