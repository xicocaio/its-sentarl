# ITS-ReLeSa

Final Masters research project implementation ITS-ReLeSA => Inteligent Trading System: a Reinforcement Learning
Sentiment Aware Approach

## Stack

The stack bellow was used mostly due to its ease of installation, configuration, and also efficiency and portability.

* Language: Python (3.7.4)
* RL Environment: OpenAI gym (0.18.3)
* RL engine: StableBaselines3 (1.0)

## Pre-installation

This system was developed in Ubuntu 16.04 but will work properly on any other Operational System(OS X, Windows, etc.).

However, this guide will only include instructions for plugins and packages that are not already installed on this OS.
For this reason, we assume that technologies like a python interpreter and pipenv are ready for use, and should not be
on the scope of this document.

* Now install pipenv dependency manager:

```bash

$ pip install --user pipenv

```

## Project configuration

Now we'll start setting up the project.

* Clone the repo from GitHub and change to project root directory. After that, install project dependencies and go to
  python virtual env, by running:

```bash
$ pipenv install
$ pipenv shell
```

## Running the project

The system will do the following:
1. Load financial data and prepare the trading environment. 2. Run a trading strategy and collect results. 3. Store all
results regarding training, testing and validation in `.csv` files. 4. Optional: Save model versions at each training
episode.

The base command to run in terminal is:

```bash
$ python app
```

There are three modes for running the project, which can be selected using the following arg

* `mode` with the allowed values: `dedault` for running a single configuration; `routine` for running several
  confiugurations; `consolidation` for grouping all results files into one single large file.

Then the arguments according to the desired modes are:

### Default

* `stg` for selecting the strategy to run. Allows: `bh` for Buy and Hold (BH); `relesa` fpr the sentiment-aware RL
  algorithm; `vanilla` for the sentiment-free version of the algorithm.
* `asset` for selecting the asset to run trade. Allows: any asset present in the `data/__init__.py` file, with the
  correspoding `.csv` file in the `data` folder.
* `algo` for selecting the RL algorithm to run. Allows: `a2c`, `ppo`, `dqn`. To add more algorithms add code to
  the `models\__init__.py` and `settings.py` files, given it is implemented by StableBaselines3.
* `epsiodes` for selecting the number of episodes to train. Allows: integer values only
* `setup` for selecting the train/val window type of setup. Allows: `static`, `rolling`.

Example to run default mode with ITS-ReLeSa using A2C

```bash
python app mode=default setup=rolling stg=relesa asset=aapl algo=A2C episodes=1 setup='rolling'
```

### Routine

* `routine_name` for selecting which routine to run when `mode` arg is set to `routine`. Allows: `default` for running
  the most basic routine, `load_model` for running routines that load a model trained up to a given episode.

Example to run default routine

```bash
python app mode=routine routine_name=default
```

> **Beware this mode can take a long time to run depending on how many assets, tcs, rolling windows, stgs and other characteristics are selected.**

### Consolidation

Example to conlidate all results files

```bash
python app mode=consolidation
```


## Folder Structure

```
its-relesa/
├── app
│   ├── common
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── utils.py
│   ├── data
│   │   ├── __init__.py
│   │   └── *.csv
│   ├── envs
│   │   ├── __init__.py
│   │   ├── exchange.py
│   │   └── stock.py
│   ├── __main__.py
│   ├── models
│   │   ├── __init__.py
│   │   └── /**/*.zip
│   ├── results
│   │   ├── __init__.py
│   │   ├── consolidated
│   │   │   └── *.csv
│   │   ├── hour
│   │   │   └── /**/*.csv
│   │   ├── monitor.py
│   │   └── organizer.py
│   ├── settings.py
│   └── setups
│       ├── base_setup.py
│       ├── __init__.py
│       ├── rolling_window.py
│       └── static.py
├── LICENSE
├── Pipfile
├── Pipfile.lock
└── README.md
```

* Root folder: Dependencies list and other metadata info.
  * `Pipfile`: Information about dependencies.
  * `README.md`: Hi.
* app: Project settings and and other general use files.
  * `settings.py`: Information regarding allowed assets, stgs and others.


## Further Improvements (TODOs)

- The utils function inside common module is starting to become to entangled with details of the system, instead of
  being self contained and separated from project particularities, maybe some of these methods should me moved elsewhere
  to other aggregation module.

- The env is calculating its rewards and returns, and its fine to some extent, however the calculation of metrics should
  probably be moved to a utils like class for general use purposes. Also, the env is doing part of the monitoring of
  results, by storing results and all. It is not completely wrong if this is being used to calculate rewards. But,
  currently it is being more used for monitoring of historic returns and actions. It could be the case that a historic
  of actions, rewards, returns and other metrics be kept in the monitor or other such class, which concern would be to
  keep track and monitor the model interaction with the env.

## Final considerations

* Go play around! =)
