This project aims at measuring the environmental cost of experimenting with metaheuristics in two typical scenarios: 1) single objective bound constrained optimization (SOBCO) and 2) evolutionary dynamic optimization (EDO). To this end we employed the framework **experiment-impact-tracker** proposed in https://github.com/Breakend/experiment-impact-tracker as follows (gray boxes):

<img src="flowchart.png" width="400">

To implement the algorithms and problems, we relied on the DEAP framework [1].

## SOBCO parameter settings

The experiments in SOBCO relied in the following parameter settings:

|             | **Parameter**               | **Settings**                                                             |
|-------------|-----------------------------|--------------------------------------------------------------------------|
|*Problems*   | Problem type                | Minimization                                                             |
|             | Dimension (D)             | <img src="https://latex.codecogs.com/svg.latex?{\in\{10,50,90,130\}}" title="{10, 50, 90, 130}"/>                                                |
|             | Objective Function          | <img src="https://latex.codecogs.com/svg.latex?{Sphere(x)}=\sum_{i=1}^{D}{x_i^2}" title="Sphere"/>|
|             |                             | <img src="https://latex.codecogs.com/svg.latex?{Rastrigin(x)}={10D}+\sum_{i=1}^{D}{x_i^2}-10\cos(2{\pi}{x_i})" title="Rastrigin"/>            |
|             |                             | <img src="https://latex.codecogs.com/svg.latex?{Rosenbrock(x)}=\sum_{i=1}^{D-1}(1-{x_i})^2+100(x_{i+1}-{x_i}^2)^2" title="Rosenbrock"/> |
|             | Search space                | <img src="https://latex.codecogs.com/svg.latex?{[-5.0,5.0]^D}" title="[-5.0, 5.0]^D"/>                                                          |
|*Algorithms* | Metaheuristic               | DE, CMAES                                                                |
|             | Generations (G)             | 500                                                                    |
|             | Population size             | 20D                                                                    |


## EDO parameter settings

|                 | **Parameter**                     | **Settings**                                  |
|-----------------|-----------------------------------|-----------------------------------------------|
| *Problems*      | Problem type                      | Maximization                                  |
|                 | Moving Peak Benchmark             | *Scenario2*                            |
|                 | Dimension (D)                   | 5                                           |
|                 | Peak function (PF)              | <img src="https://latex.codecogs.com/svg.latex?{Cone(x)}=\sqrt{\sum_{i}^D({x_{i}^p}-{x_i})^2}" title="Cone"/> |
|                 | Search space                      | <img src="https://latex.codecogs.com/svg.latex?{[0.0,100]^D}" title="[0.0, 100]^D"/>                                |
|                 | Number of peaks (Peaks)         | <img src="https://latex.codecogs.com/svg.latex?{\in\{10,20,30,40\}}" title="{10, 20, 30, 40}"/>                      |
|                 | Number of changes (Changes)     | 100                                         |
|                 | Change severity (s)             | 1.0                                         |
|                 | Change frequency (CF)           | <img src="https://latex.codecogs.com/svg.latex?{\in\{2500,5000,7500,10000\}}" title="{2500, 5000, 7500, 10000}"/>             |
|*Algorithms*     | Metaheuristic                     | mQSO                                        |
|                 | Number of swarms ($Swarms$)       | <img src="https://latex.codecogs.com/svg.latex?{\in\{1,10,20,30\}}" title="{1, 10, 20, 30}"/>                       |
|                 | Number of neutral particles (n) | 5                                           |
|                 | Number of quantum particles (q) | 5                                           |
|                 | Quantum radius                  | 0.5                                           |



As for the execution environment we used the following settings for both set of experiments:

| **Parameter**               | **Settings**                                                             |
|-----------------------------|--------------------------------------------------------------------------|
| Runs                        | 30                                                                       |
| Random seed                 | <img src="https://latex.codecogs.com/svg.latex?{\in\{1,2,{...},30\}}" title="{1,2,...,30}"/>                                                 |
| Performance measure SOBCO   | Best fitness                                                             |
| Performance measure EDO     | Offline error                                                            |
| Computer                    | iMac (Retina 5K, 27-inch, 2019)                                          |
| Operating System            | macOS Big Sur 11.1                                                       |
| CPU                         | Intel(R) Core(TM) i5-8500 CPU @ 3.00GHz                                  |
| RAM                         | 40 GB 2667 MHz DDR4                                                      |


# Results

## SOBCO

<img src="results/total_energy_consumption_sobco.png" width="600">

<img src="results/total_carbon_emissions_sobco.png" width="600">

<img src="results/performance_sobco.png" width="600">

## EDO

<img src="results/total_energy_consumption_edo.png" width="600">

<img src="results/total_carbon_emissions_edo.png" width="600">

<img src="results/performance_edo.png" width="600">

# Social cost by country

<img src="results/scenarios_comparison.png" width="600">




# References

[1] F.-A. Fortin, F.-M. D. Rainville, M.-A. Gardner,M. Parizeau, C. Gagné, Deap: Evolutionary algorithms made easy, Journal of Machine LearningResearch 13 (70) (2012) 2171–2175
