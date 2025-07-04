# SUBJECT 2: Optimization for the Optimal Sequencing of Manufacturing Orders in a Factory

## a) Project Description

This project aims to develop an optimization algorithm to efficiently sequence manufacturing orders in a cork factory (DIAM).  
The objective is to minimize delivery delays, simplify production planning, and allow for dynamic adjustments in case of unforeseen events (e.g., machine breakdowns, urgent new orders, etc.).

## b) Technical Choices
- **Sets**
  - I represents the set of orders
  - M represents the set of machines
  - T represents the set of time slots

- **Variables**
  - $x(i,m,t)=1$ if the task i is being done on the machine m at the time t
  - $x(i,m,t) =0$ otherwise
(This variable registers all the necessary information to build the OF time schedule.)
  - $t_{\text{early}}(i)$ represents the advance with which the order i is ended
  - $t_{\text{early}}(i)$ represents the advance with which the order i is ended
  - $\text{nb}_{\text{changes}}$ represents the number of machine changes
  - \( k_1 = 1 \)
  - \( k_2 = 10 \)
  - \( k_3 = 1000 \)
  -  \( k_4 = 100 \)

  > Note: These values were chosen arbitrarily and can be easily adjusted as needed.
- **Cost function**:

$$
  \text{Cost} = k_1 \sum_i t_{\text{early}}(i)^2 + k_2 \sum_i t_{\text{late}}(i)^2 + k_3 \cdot \text{nb}_{\text{changes}} + k_4 . surplus
  $$

  Where:
  - $i$ represents the index of the order
  - $t_{\text{early}}(i)$ represents the advance with which the order i is ended
  - $t_{\text{early}}(i)$ represents the advance with which the order i is ended
  - $\text{nb}_{\text{changes}}$ represents the number of machine changes
  - \( k_1 = 1 \)
  - \( k_2 = 10 \)
  - \( k_3 = 1000 \)
  -  \( k_4 = 100 \)

  > Note: These values were chosen arbitrarily and can be easily adjusted as needed.
- **Constraints**:
  - Only one manufacturing order (OF) can be processed at a time on each machine.
    It can be mathematically represented as: $\forall (m,t) \in M\times T $ $\sum_i x(i,m,t)$ $\leq$ $1$
  - A maximum of two machines can work on the same OF simultaneously, due to the availability of only two printing supports per OF.
    It can be mathematically represented as: $\forall (i,t) \in I\times T $ $\sum_m x(i,m,t)$ $\leq$ number of stamps
  - Double-printed corks can be printed either once on a dual-head machine or twice on a single-head machine.
  

- **Simplifying Assumptions**:
  - Time has been discretized using a 30-minute time step. Note: this step size can be easily adjusted if needed.
  - A constant changeover time of 30 minutes is assumed between two different OFs.
    
## c) Required Tools and Libraries

We used **Gurobi** as the optimization solver (with a free academic license).  
The following Python libraries are also required:  
  -matplotlib  
  -pandas  
  -datetime  
  

## d) Stable Version Commit

The stable and validated version of this project corresponds to the following commit hash:

{insert hash}

