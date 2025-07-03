# SUBJECT 2: Optimization for the Optimal Sequencing of Manufacturing Orders in a Factory

## a) Project Description

This project aims to develop an optimization algorithm to efficiently sequence manufacturing orders in a factory.  
The objective is to minimize delivery delays, simplify production planning, and allow for dynamic adjustments in case of unforeseen events (e.g., machine breakdowns, urgent new orders, etc.).

## b) Technical Choices

- **Cost function**:

  $$
  \text{Cost} = k_1 \sum_i t_{\text{early}}(i)^2 + k_2 \sum_i t_{\text{late}}(i)^2 + k_3 \cdot \text{nb}_{\text{changes}}
  $$

- **Constraints**:
  - Only one manufacturing order (OF) can be processed at a time on each machine.
  - Double-printed corks can be printed either once on a dual-head machine or twice on a single-head machine.
  - A maximum of two machines can work on the same OF simultaneously, due to the availability of only two printing supports per OF.

- **Simplifying Assumptions**:
  - Time has been discretized using a 30-minute time step. Note: this step size can be easily adjusted if needed.
  - A constant changeover time of 30 minutes is assumed between two different OFs.

## c) Required Tools and Libraries

We used **Gurobi** as the optimization solver (with a free academic license).  
The following Python libraries are also required:

```bash
matplotlib
pandas
datetime
gurobipy

