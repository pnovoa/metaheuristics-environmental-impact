import argparse
import sys
from compute_tracker import ImpactTracker
from data_interface import DataInterface

import numpy as np

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools
import pandas as pd
import json
from datetime import datetime
import random
import array
from itertools import chain
import math

data_base = pd.read_csv('cscc_db_v2.csv')
hof = tools.HallOfFame(1)


class ParameterSettings:
    def __init__(self, argparse_name_space) -> object:
        self.arg_name_space = argparse_name_space
        self.run = argparse_name_space.run
        self.search_space_dimension = argparse_name_space.func_dim
        self.population_size = argparse_name_space.algo_psize
        self.generations = argparse_name_space.gen
        self.function_to_minimize = getattr(benchmarks, argparse_name_space.func)
        if argparse_name_space.algo == "cmaes":
            self.algorithm_to_run = run_cmaes
        elif argparse_name_space.algo == "de":
            self.algorithm_to_run = run_de
        else:
            raise Exception(
                "No algorithm named as {} was found. Please try with another name".format(argparse_name_space.algo))

        if argparse_name_space.output_folder is None:
            self.output_folder = "output"
        else:
            self.output_folder = argparse_name_space.output_folder

        self.random_seed = self.run


def read_parameters() -> ParameterSettings:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--algo_psize", type=int, required=False, default=100)
    parser.add_argument('--func', type=str, required=True)
    parser.add_argument('--func_dim', type=int, required=False, default=30)
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--gen', type=int, required=False, default=250)
    parser.add_argument('--output_folder', type=str, required=False, default=None)
    args = parser.parse_args()
    return ParameterSettings(args)


def run_cmaes(parameter_settings) -> tools.Logbook:
    hof = tools.HallOfFame(1)
    d = parameter_settings.search_space_dimension
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("evaluate", parameter_settings.function_to_minimize)

    # The cma module uses the numpy random number generator
    np.random.seed(parameter_settings.random_seed)

    # The CMA-ES algorithm takes a population of one individual as argument
    # The centroid is set to a vector of 5.0 see http://www.lri.fr/~hansen/cmaes_inmatlab.html
    # for more details about the rastrigin and other tests for CMA-ES
    strategy = cma.Strategy(centroid=[5.0] * d, sigma=5.0, lambda_=20 * d)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    # hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # The CMA-ES algorithm converge with good probability with those settings
    _, logbook = algorithms.eaGenerateUpdate(toolbox,
                                             ngen=parameter_settings.generations,
                                             stats=stats,
                                             halloffame=hof,
                                             verbose=False)
    return hof, logbook


def run_de(parameter_settings):
    NDIM = parameter_settings.search_space_dimension
    hof = tools.HallOfFame(1)
    random.seed(parameter_settings.random_seed)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

    def mutDE(_y, _a, _b, _c, f):
        size = len(_y)
        for _i in range(size):
            _y[_i] = _a[_i] + f * (_b[_i] - _c[_i])
        return _y

    def cxBinomial(_x, _y, cr):
        size = len(_x)
        index = random.randrange(size)
        for _i in range(size):
            if _i == index or random.random() < cr:
                _x[_i] = _y[_i]
        return _x

    def cxExponential(_x, _y, cr):
        size = len(_x)
        index = random.randrange(size)
        # Loop on the indices index -> end, then on 0 -> index
        for _i in chain(range(index, size), range(0, index)):
            _x[_i] = _y[_i]
            if random.random() < cr:
                break
        return _x

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -6., 6.)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, NDIM)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mutate", mutDE, f=0.8)
    toolbox.register("mate", cxExponential, cr=0.8)
    toolbox.register("select", tools.selRandom, k=3)
    toolbox.register("evaluate", parameter_settings.function_to_minimize)

    # Differential evolution parameters
    mu = NDIM * 20
    ngen = parameter_settings.generations

    pop = toolbox.population(n=mu)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "nevals", "avg", "std", "min", "max"

    # Evaluate the individuals
    fitness_list = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitness_list):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    # print(logbook.stream)

    for g in range(1, ngen):
        children = []
        for agent in pop:
            # We must clone everything to ensure independence
            a, b, c = [toolbox.clone(ind) for ind in toolbox.select(pop)]
            x = toolbox.clone(agent)
            y = toolbox.clone(agent)
            y = toolbox.mutate(y, a, b, c)
            z = toolbox.mate(x, y)
            del z.fitness.values
            children.append(z)

        fitness_list = toolbox.map(toolbox.evaluate, children)
        for (i, ind), fit in zip(enumerate(children), fitness_list):
            ind.fitness.values = fit
            if ind.fitness > pop[i].fitness:
                pop[i] = ind

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=g, nevals=len(pop), **record)

    # print("Best individual is ", hof[0])
    # print("with fitness", hof[0].fitness.values[0])
    return hof, logbook


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
    ssc = ssc[np.isnan(ssc["dr"])]  # use only growth adjusted models
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


def main():
    header_to_save = ["utc_date_time",
                      "algorithm",
                      "function",
                      "dimension",
                      "generations",
                      "evaluations",
                      "seed",
                      "run",
                      "best_fitness",
                      "kg_carbon",
                      "total_power",
                      "pue",
                      "country_code",
                      "median_carbon_cost",
                      "lower_carbon_cost",
                      "upper_carbon_cost"
                      ]

    #df_save = pd.DataFrame(columns=header_to_save)

    with open("countries_to_report.json", "r") as json_file:
        list_countries = json.load(json_file)

    execution_name = ' '.join(sys.argv[1:])

    print("STARTING execution {}\n".format(execution_name))
    utc_datetime = datetime.utcnow()
    utc_datetime_ts = str(datetime.timestamp(utc_datetime))
    par_set = read_parameters()
    args = par_set.arg_name_space

    tracker_log_folder = "{}/{}".format(par_set.output_folder, utc_datetime_ts)
    # try:
    # Impact tracker
    tracker = ImpactTracker(tracker_log_folder)
    tracker.launch_impact_monitor()

    hof, logbook_performance = par_set.algorithm_to_run(par_set)

    tracker.stop()
    # Saving the algorithm results
    df_log = pd.DataFrame(logbook_performance)
    # df_log.to_csv('{}/{}.pge'.format(par_set.output_folder, utc_datetime_ts), index=False)
    # Reading the tracker info
    impact_data = DataInterface(tracker_log_folder)

    fitness_evaluations = df_log['nevals'].iloc[-1]
    best_fitness = hof[0].fitness.values[0]

    data_to_save = [str(utc_datetime),
                    args.algo,
                    args.func,
                    args.func_dim,
                    args.gen,
                    fitness_evaluations,
                    par_set.random_seed,
                    par_set.run,
                    best_fitness,
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
