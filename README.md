# Swarm Intelligence
We provide Python implementations of swarm-based optimization algorithms for mining gradual patterns. The algorithm implementations include:

* Genetic Algorithm
* Particle Swarm Optimization
* Wasp Swarm Optimization
* Pure Random Search
* Pure Local Search

### Requirements:
You will be required to install the following Python dependencies:

```
                   install python (version => 3.6)

```

### Usage:
Use it a command line program with the local package to mine gradual patterns:

For example we executed the <em><strong>GA</strong>-GRAANK</em> algorithm on a sample data-set

```
$python3 src/pkg_main/main.py -a 'ga' -f data/DATASET.csv
```

where you specify the input parameters as follows:

* <strong>algorithm</strong> - [required] select algorithm ```ga, pso, wso, prs, pls```
* <strong>filename.csv</strong> - [required] a file in csv format
* <strong>minSup</strong> - [optional] minimum support ```default = 0.5```

### License:
* MIT

### References
* Dickson Owuor, Anne Laurent, and Joseph Orero (2019). Mining Fuzzy-temporal Gradual Patterns. In the proceedings of the 2019 IEEE International Conference on Fuzzy Systems (FuzzIEEE). IEEE. https://doi.org/10.1109/FUZZ-IEEE.2019.8858883.
* Owuor, D., Runkler T., Laurent A., Menya E., Orero J (2021), Ant Colony Optimization for Mining Gradual Patterns. International Journal of Machine Learning and Cybernetics.
* Anne Laurent, Marie-Jeanne Lesot, and Maria Rifqi. 2009. GRAANK: Exploiting Rank Correlations for Extracting Gradual Itemsets. In Proceedings of the 8th International Conference on Flexible Query Answering Systems (FQAS '09). Springer-Verlag, Berlin, Heidelberg, 382-393.
