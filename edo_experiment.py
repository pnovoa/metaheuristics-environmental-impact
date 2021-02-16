import argparse
import sys
from compute_tracker import ImpactTracker
from data_interface import DataInterface

import numpy
import operator

from deap import base
from deap import benchmarks
from deap.benchmarks import movingpeaks
from deap import creator
from deap import tools
import pandas as pd
import json
from datetime import datetime
import random
import itertools
import math

scenario = movingpeaks.SCENARIO_2
data_base = pd.read_csv('cscc_db_v2.csv')
hof = tools.HallOfFame(1)


def read_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_swarms", type=int, required=True)
    parser.add_argument("--n_peaks", type=int, required=True)
    parser.add_argument("--n_changes", type=int, required=False, default=100)
    parser.add_argument("--change_freq", type=int, required=True)
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--output_folder', type=str, required=False, default="output_dynamic")
    args = parser.parse_args()
    return args


def read_impacts(kg_carbon, countries) -> list:
    data_rows = []
    for c in countries:
        data_country = read_impact_by_country(kg_carbon, c)
        data_rows.append(data_country)
    return data_rows


def read_impact_by_country(kg_carbon, country_code) -> list:
    ssc = data_base

    # only use short-run model
    ssc = ssc[ssc["run"] == "bhm_sr"]
    ssc = ssc[ssc["SSP"] == "SSP2"]
    ssc = ssc[ssc["ISO3"] == country_code]
    ssc = ssc[numpy.isnan(ssc["dr"])]  # use only growth adjusted models
    ssc = ssc[ssc["prtp"] == 2]  # a growth adjusted discount rate with 2% pure rate of time preference
    ssc = ssc[ssc["eta"] == "1p5"]  # IES of 1.5
    ssc = ssc[ssc["RCP"] == "rcp60"]  # rcp 6, middle of the road
    ssc = ssc[ssc["dmgfuncpar"] == "bootstrap"]
    ssc = ssc[ssc["climate"] == "uncertain"]

    median = ssc["50%"]
    lower = ssc["16.7%"]
    upper = ssc["83.3%"]

    median_carbon_cost = (kg_carbon / 1000.) * float(median)
    upper_carbon_cost = (kg_carbon / 1000.) * float(upper)
    lower_carbon_cost = (kg_carbon / 1000.) * float(lower)

    return [country_code, median_carbon_cost, lower_carbon_cost, upper_carbon_cost]


def run_pso_multi_swarm_on_moving_peaks(parameter_settings):
    NDIM = 5
    NSWARMS = parameter_settings.n_swarms

    MAX_FES = parameter_settings.n_changes * parameter_settings.change_freq

    NPARTICLES = 5
    NEXCESS = 3
    RCLOUD = 0.5  # 0.5 times the move severity

    _scenario = movingpeaks.SCENARIO_2

    _scenario["npeaks"] = parameter_settings.n_peaks
    _scenario["period"] = parameter_settings.change_freq

    BOUNDS = [_scenario["min_coord"], _scenario["max_coord"]]

    mpb = movingpeaks.MovingPeaks(dim=NDIM, **_scenario)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
                   best=None, bestfit=creator.FitnessMax)
    creator.create("Swarm", list, best=None, bestfit=creator.FitnessMax)

    def generate(pclass, dim, pmin, pmax, smin, smax):
        part = pclass(random.uniform(pmin, pmax) for _ in range(dim))
        part.speed = [random.uniform(smin, smax) for _ in range(dim)]
        return part

    def convertQuantum(_swarm, rcloud, centre, dist):
        dim = len(_swarm[0])
        for part in _swarm:
            position = [random.gauss(0, 1) for _ in range(dim)]
            dist = math.sqrt(sum(x ** 2 for x in position))

            if dist == "gaussian":
                u = abs(random.gauss(0, 1.0 / 3.0))
                part[:] = [(rcloud * x * u ** (1.0 / dim) / dist) + c for x, c in zip(position, centre)]

            elif dist == "uvd":
                u = random.random()
                part[:] = [(rcloud * x * u ** (1.0 / dim) / dist) + c for x, c in zip(position, centre)]

            elif dist == "nuvd":
                u = abs(random.gauss(0, 1.0 / 3.0))
                part[:] = [(rcloud * x * u / dist) + c for x, c in zip(position, centre)]

            del part.fitness.values
            del part.bestfit.values
            part.best = None

        return _swarm

    def updateParticle(part, best, chi, c):
        ce1 = (c * random.uniform(0, 1) for _ in range(len(part)))
        ce2 = (c * random.uniform(0, 1) for _ in range(len(part)))
        ce1_p = map(operator.mul, ce1, map(operator.sub, best, part))
        ce2_g = map(operator.mul, ce2, map(operator.sub, part.best, part))
        a = map(operator.sub,
                map(operator.mul,
                    itertools.repeat(chi),
                    map(operator.add, ce1_p, ce2_g)),
                map(operator.mul,
                    itertools.repeat(1 - chi),
                    part.speed))
        part.speed = list(map(operator.add, part.speed, a))
        part[:] = list(map(operator.add, part, part.speed))

    toolbox = base.Toolbox()
    toolbox.register("particle", generate, creator.Particle, dim=NDIM,
                     pmin=BOUNDS[0], pmax=BOUNDS[1], smin=-(BOUNDS[1] - BOUNDS[0]) / 2.0,
                     smax=(BOUNDS[1] - BOUNDS[0]) / 2.0)
    toolbox.register("swarm", tools.initRepeat, creator.Swarm, toolbox.particle)
    toolbox.register("update", updateParticle, chi=0.729843788, c=2.05)
    toolbox.register("convert", convertQuantum, dist="nuvd")
    toolbox.register("evaluate", mpb)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "nswarm", "evals", "error", "offline_error", "avg", "max"

    # Generate the initial population
    population = [toolbox.swarm(n=NPARTICLES) for _ in range(NSWARMS)]

    # Evaluate each particle
    for swarm in population:
        for part in swarm:
            part.fitness.values = toolbox.evaluate(part)

            # Update swarm's attractors personal best and global best
            if not part.best or part.fitness > part.bestfit:
                part.best = toolbox.clone(part[:])  # Get the position
                part.bestfit.values = part.fitness.values  # Get the fitness
            if not swarm.best or part.fitness > swarm.bestfit:
                swarm.best = toolbox.clone(part[:])  # Get the position
                swarm.bestfit.values = part.fitness.values  # Get the fitness

    record = stats.compile(itertools.chain(*population))
    logbook.record(gen=0, evals=mpb.nevals, nswarm=len(population),
                   error=mpb.currentError(), offline_error=mpb.offlineError(), **record)

    #    if verbose:
    #        print(logbook.stream)

    generation = 1
    while mpb.nevals < MAX_FES:
        # Check for convergence
        rexcl = (BOUNDS[1] - BOUNDS[0]) / (2 * len(population) ** (1.0 / NDIM))

        not_converged = 0
        worst_swarm_idx = None
        worst_swarm = None
        for i, swarm in enumerate(population):
            # Compute the diameter of the swarm
            for p1, p2 in itertools.combinations(swarm, 2):
                d = math.sqrt(sum((x1 - x2) ** 2. for x1, x2 in zip(p1, p2)))
                if d > 2 * rexcl:
                    not_converged += 1
                    # Search for the worst swarm according to its global best
                    if not worst_swarm or swarm.bestfit < worst_swarm.bestfit:
                        worst_swarm_idx = i
                        worst_swarm = swarm
                    break

        # If all swarms have converged, add a swarm
        if not_converged == 0:
            population.append(toolbox.swarm(n=NPARTICLES))
        # If too many swarms are roaming, remove the worst swarm
        elif not_converged > NEXCESS:
            population.pop(worst_swarm_idx)

        # Update and evaluate the swarm
        for swarm in population:
            # Check for change
            if swarm.best and toolbox.evaluate(swarm.best) != swarm.bestfit.values:
                # Convert particles to quantum particles
                swarm[:] = toolbox.convert(swarm, rcloud=RCLOUD, centre=swarm.best)
                swarm.best = None
                del swarm.bestfit.values

            for part in swarm:
                # Not necessary to update if it is a new swarm
                # or a swarm just converted to quantum
                if swarm.best and part.best:
                    toolbox.update(part, swarm.best)
                part.fitness.values = toolbox.evaluate(part)

                # Update swarm's attractors personal best and global best
                if not part.best or part.fitness > part.bestfit:
                    part.best = toolbox.clone(part[:])
                    part.bestfit.values = part.fitness.values
                if not swarm.best or part.fitness > swarm.bestfit:
                    swarm.best = toolbox.clone(part[:])
                    swarm.bestfit.values = part.fitness.values

        record = stats.compile(itertools.chain(*population))
        logbook.record(gen=generation, evals=mpb.nevals, nswarm=len(population),
                       error=mpb.currentError(), offline_error=mpb.offlineError(), **record)

        #        if verbose:
        #            print(logbook.stream)

        # Apply exclusion
        reinit_swarms = set()
        for s1, s2 in itertools.combinations(range(len(population)), 2):
            # Swarms must have a best and not already be set to reinitialize
            if population[s1].best and population[s2].best and not (s1 in reinit_swarms or s2 in reinit_swarms):
                dist = 0
                for x1, x2 in zip(population[s1].best, population[s2].best):
                    dist += (x1 - x2) ** 2.
                dist = math.sqrt(dist)
                if dist < rexcl:
                    if population[s1].bestfit <= population[s2].bestfit:
                        reinit_swarms.add(s1)
                    else:
                        reinit_swarms.add(s2)

        # Reinitialize and evaluate swarms
        for s in reinit_swarms:
            population[s] = toolbox.swarm(n=NPARTICLES)
            for part in population[s]:
                part.fitness.values = toolbox.evaluate(part)

                # Update swarm's attractors personal best and global best
                if not part.best or part.fitness > part.bestfit:
                    part.best = toolbox.clone(part[:])
                    part.bestfit.values = part.fitness.values
                if not population[s].best or part.fitness > population[s].bestfit:
                    population[s].best = toolbox.clone(part[:])
                    population[s].bestfit.values = part.fitness.values

        generation += 1

    return logbook


def main():
    header_to_save = ["utc_date_time",
                      "algorithm",
                      "function",
                      "dimension",
                      "generations",
                      "evaluations",
                      "seed",
                      "run",
                      "mean_offline_error",
                      "kg_carbon",
                      "total_power",
                      "pue",
                      "country_code",
                      "median_carbon_cost",
                      "lower_carbon_cost",
                      "upper_carbon_cost"
                      ]

    # df_save = pd.DataFrame(columns=header_to_save)

    with open("countries_to_report.json", "r") as json_file:
        list_countries = json.load(json_file)

    execution_name = ' '.join(sys.argv[1:])

    print("STARTING execution {}\n".format(execution_name))
    utc_datetime = datetime.utcnow()
    utc_datetime_ts = str(datetime.timestamp(utc_datetime))
    par_set = read_parameters()
    random.seed(par_set.run)
    numpy.random.seed(par_set.run + 1)

    tracker_log_folder = "{}/{}".format(par_set.output_folder, utc_datetime_ts)
    # try:
    # Impact tracker
    tracker = ImpactTracker(tracker_log_folder)
    tracker.launch_impact_monitor()

    logbook_performance = run_pso_multi_swarm_on_moving_peaks(par_set)

    tracker.stop()
    # Saving the algorithm results
    df_log = pd.DataFrame(logbook_performance)
    # df_log.to_csv('{}/{}.pge'.format(par_set.output_folder, utc_datetime_ts), index=False)
    # Reading the tracker info
    impact_data = DataInterface(tracker_log_folder)

    data_to_save = [str(utc_datetime),
                    "mQSO",
                    "cone",
                    5,
                    df_log['gen'].iloc[-1],
                    df_log['evals'].iloc[-1],
                    par_set.run,
                    par_set.run,
                    df_log['offline_error'].mean(),
                    impact_data.total_power,
                    impact_data.kg_carbon,
                    impact_data.PUE]

    # df_run = pd.DataFrame([data_to_save], columns=header_to_save)
    # df_run.to_csv('{}/{}.pru'.format(par_set.output_folder, utc_datetime_ts), index=False)

    data_to_save_countries = []
    data_countries = read_impacts(impact_data.kg_carbon, list_countries["countries"])
    for row2 in data_countries:
        data_to_save_countries.append(data_to_save[:] + row2)

    df_save = pd.DataFrame(data_to_save_countries, columns=header_to_save)
    # df_save = df_save.append(df_social_cost_run, ignore_index=True)

    print("\nFINISH execution {} successfully!\n".format(execution_name))

    df_save.to_csv('{}/{}.csc'.format(par_set.output_folder, utc_datetime_ts), index=False)


if __name__ == '__main__':
    main()
