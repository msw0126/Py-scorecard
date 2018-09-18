
from Learn.TrainRobot import train_robot

def learn(data, dictionary, target_varname, learn_dir, metric = "ks", algorithm = "glmnet",
            quantiles = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1],
            sample_miss_cutoff = 0.95, variable_miss_cutoff = 0.95, variable_zero_cutoff = 0.95,
            max_nFactorLevels = 1024, nbins = 10, truncation_cutoff = 20, breaks_zero_cutoff = 0.3,
            iv_cutoff = 0.01, collinearity_cutoff = 0.9, unbalanced_cutoff = 3, onehot = True):

    train_robot(data, dictionary, target_varname, learn_dir, metric, algorithm,
                quantiles, sample_miss_cutoff, variable_miss_cutoff, variable_zero_cutoff,
                max_nFactorLevels, nbins, truncation_cutoff, breaks_zero_cutoff,
                iv_cutoff, collinearity_cutoff, unbalanced_cutoff)