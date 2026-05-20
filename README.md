# Human Hippocampal Replay as Search
This is the official code repository of the paper titled "Human Hippocampal Replay as Search" presented at CogSci 2026. 

## Overview
[Schwartenbeck, Philipp, et al. "Generative replay underlies compositional inference in the hippocampal-prefrontal circuit." Cell 186.22 (2023): 4885-4897.](https://www.cell.com/neuron/fulltext/S0896-6273(22)01125-4?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0896627322011254%3Fshowall%3Dtrue) present a task where humans display forward hippocampal replay sequences indicative of parallel and structured hypothesis testing. 


<p align="center">
<img width="30%" src="https://github.com/mishaalkandapath/hpp_search/blob/main/media/sample_game_response.png">
</p>

Models of replay commonly cast replay as a sequential non-parallel process. We presenet a new model based off a popular one in [Jensen et al. "A recurrent network model of planning explains hippocampal replay and human behavior". Nature Neuroscience (2024).](https://www.nature.com/articles/s41593-024-01675-7) to architecturally facilitate parallel search.  A brief schematic of the model is shown below.

<p align="center">
<img width="50%" src="https://github.com/mishaalkandapath/hpp_search/blob/main/media/403model.png">
</p>

### Results

### Performance
All variants attained similar correctness and time-to-criterion on the training set. Most reached the threshold criterion on the test set. However, all _full planning_ (n=5) variants reached the performance threshold substantially earlier (median = 614) than their _baseline_ (n=5) counterparts (median = 981), with complete separation between groups (𝑟𝑟𝑏 = 1.0). A Mann-Whitney test was significant (𝑝 < 0.01), suggesting of a learning-speed advantage for the full planning models. 

### Neural Replay Sequences 

Model transitions averaged during period of high search entropy were qualitatively similar to human trajectories (evidence of structured hypothesis testing):

<p align="center">
<img width="50%" src="https://github.com/mishaalkandapath/hpp_search/blob/main/media/sequenceness.png">
</p>

### High Search Entropy
An interesting pattern emerges across planning variants, and especially in full-planning variants. Performance curves initially plateau. Then a sudden increase in search entropy and a sudden decrease in performance is observed, which is immediately followed by a reduction in search entropy and a jump in generalization. 

<p align="center">
<img width="50%" src="https://github.com/mishaalkandapath/hpp_search/blob/main/media/logperformance.png">
</p>

This pattern was unexpected, but closely resembles findings from [Gray and Lindstedt. "Plateaus, Dips, and Leaps: Where to Look for Inventions and Discoveries During Skilled Performance" (2017).](https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.12412). They describe similar dynamics in human skill learning, where short-term declines in performance often precede major improvements. These declines tend to occur during periods of experimentation and strategy discov- ery, which later produce jumps in performance. Our results may provide a mechanistic explanation of improvements in skill accumulation given models that architecturally allow for search.
