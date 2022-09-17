# ITS-SentARL

***Intelligent Trading Systems: A Sentiment-Aware Reinforcement Learning Approach*** research project implementation used in:

1. Second ACM International Conference on AI in Finance (ICAIF'21) article
    - [ACM Official Version](https://dl.acm.org/doi/10.1145/3490354.3494445)
    - [Free pre-print version (arXiv)](https://arxiv.org/abs/2112.02095)
2. [F. C. Lima Paiva](https://www.linkedin.com/in/xicocaio/) (aka @xicocaio) Master's Thesis.



![ITS-SentARL general architecture. Image source: Intelligent Trading Systems: A Sentiment-Aware Reinforcement Learning Approach presented in the Second ACM International Conference on AI in Finance (ICAIF'21)](docs/\_static/img/general_architecture_line_v2_image.png?raw=true "ITS-SentARL general architecture. Image source: Intelligent Trading Systems: A Sentiment-Aware Reinforcement Learning Approach presented in the Second ACM International Conference on AI in Finance (ICAIF'21)")


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
  configurations; `consolidation` for grouping all results files into one single large file.

Then the arguments according to the desired modes are:

### Default

* `stg` for selecting the strategy to run. Allows: `bh` for Buy and Hold (BH); `sentarl` fpr the sentiment-aware RL
  algorithm; `vanilla` for the sentiment-free version of the algorithm.
* `asset` for selecting the asset to run trade. Allows: any asset present in the `data/__init__.py` file, with the
  corresponding `.csv` file in the `data` folder.
* `algo` for selecting the RL algorithm to run. Allows: `a2c`, `ppo`, `dqn`. To add more algorithms add code to
  the `models\__init__.py` and `settings.py` files, given it is implemented by StableBaselines3.
* `epsiodes` for selecting the number of episodes to train. Allows: integer values only
* `setup` for selecting the train/val window type of setup. Allows: `static`, `rolling`.

Example to run default mode with ITS-SentARL using A2C

```bash
python app mode=default setup=rolling stg=sentarl asset=aapl algo=A2C episodes=1 setup='rolling'
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

Example to consolidate all results files

```bash
python app mode=consolidation
```


## Folder Structure

```
its-sentarl/
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
* app: Project settings and other general use files.
  * `settings.py`: Information regarding allowed assets, stgs and others.


## Citing the Project

To cite this repository in publications, please use the following bibtex formated text:

```bibtex
@inproceedings{LimaPaiva2021,
    author = {{Lima Paiva}, Francisco Caio and Felizardo, Leonardo Kanashiro and Bianchi, Reinaldo Augusto da Costa Bianchi and Costa, Anna Helena Reali},
    title = {Intelligent Trading Systems: A Sentiment-Aware Reinforcement Learning Approach},
    year = {2021},
    isbn = {9781450391481},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3490354.3494445},
    doi = {10.1145/3490354.3494445},
    booktitle = {Proceedings of the Second ACM International Conference on AI in Finance},
    articleno = {40},
    numpages = {9},
    keywords = {deep reinforcement learning, sentiment analysis, stock markets},
    location = {Virtual Event},
    series = {ICAIF '21},
    archivePrefix = {arXiv},
    eprint = {2112.02095},
    primaryClass = {q-fin.TR}
}
```

## Suplementary Material and Components

Reading:
- [Full Article (ACM)](https://dl.acm.org/doi/10.1145/3490354.3494445)
- [pre-print version (arXiv)](https://arxiv.org/abs/2112.02095)

Sentiment extractor module details:
- [Article](https://www.researchgate.net/publication/339962669_Assessing_Regression-Based_Sentiment_Analysis_Techniques_in_Financial_Texts)
- [Source code](https://bit.ly/3kzau8G)

Financial news webcrawler:
- [Source code](https://github.com/xicocaio/financial_web_crawler)

## Acknowledgments

This work was financed in part by Itaú Unibanco S.A. through the Programa de Bolsas Itaú (PBI) of the Centro de Ciência de Dados ([C2D](http://c2d.poli.usp.br/)), EP-USP), by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES Finance Code 001), Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq  grant 310085/2020-9 and 88882.333380/2019-01),and by the Center for Artificial Intelligence (C4AI-USP), with support from FAPESP (grant 2019/07665-4) and \textit{IBM Corporation}.
Any views and opinions expressed in this article are those of the authors and do not necessarily reflect the official policy or position of the funding companies.

## Final considerations

* Go play around! =)
