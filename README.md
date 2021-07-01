# its-relesa
Final Masters research project implementation ITS-ReLeSA => Inteligent Trading System: a Reinforcement Learning Sentiment Aware Approach

## TODOs
- The utils function inside common module is starting to become to entangled with details of the system,
  instead of being self contained and separated from project particularities, maybe some of these methods
  should me moved elsewhere to other aggregation module.
  
- The env is calculating its rewards and returns, and its fine to some extent, however the calculation of metrics should
  probably be moved to a utils like class for general use purposes. Also, the env is doing part of the monitoring
  of results, by storing results and all. It is not completely wrong if this is being used to calculate rewards. But,
  currently it is being more used for monitoring of historic returns and actions. It could be the case that a historic
  of actions, rewards, returns and other metrics be kept in the monitor or other such class, which concern would be
  to keep track and monitor the model interaction with the env.