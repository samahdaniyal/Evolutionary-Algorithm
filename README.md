# Evolutionary-Algorithm

**CS 451 - Computational Intelligence, Spring 2025, Habib University**  

---

## Overview

This repository contains the implementation of an Evolutionary Algorithm (EA) applied to solve several complex optimization problems. The assignment explores how EA techniques can be leveraged to tackle the following challenges:
- **Travelling Salesman Problem (TSP)**
- **Job-Shop Scheduling Problem (JSSP)**
- **Evolutionary Art**

The project illustrates various aspects of EA including problem formulation, chromosome representation, selection mechanisms, crossover, mutation, survivor selection, and algorithm execution. It also includes detailed analysis and results, demonstrating how different EA parameters impact convergence and solution quality.

---

## Objective

The primary goal of this assignment is to provide insight into global optimization using Evolutionary Algorithms. By mapping real-world problems to computational models, the project demonstrates how various EA parameters and strategies—such as selection schemes, crossover, and mutation—affect performance and convergence towards an optimal solution.

---

## Problem Formulation

### Travelling Salesman Problem (TSP)

The TSP aims to find the shortest possible route that visits each city exactly once and returns to the origin. In this assignment, the Qatar dataset (194 cities) is used with a known optimal tour length of 9352.

---

### Job-Shop Scheduling Problem (JSSP)

The JSSP is an NP-hard problem aimed at finding the optimal schedule for jobs across shared resources to minimize total completion time.

---

### Evolutionary Art

Evolutionary Art uses EA to generate images that increasingly resemble a target image.

---

## Dependencies

Python 3.8+
NumPy
Matplotlib

---

## References:

The Qatar TSP dataset (qa194.tsp) was obtained from [University of Waterloo](https://www.math.uwaterloo.ca/tsp/world/countries.html).
The JSSP data (abz7, la19, ft10) was taken from [here](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt)

