import pandas as pd

def bootstrap_statistic(df, function,  bootstrap_samples=1000, lower_confidence=0.025, upper_confidence=0.975, values=False):
    """ This gives bootstrap confidence intervals on the population value
        of function given the sample represented by iterable."""
    treated_outcome = []
    control_outcome = []
    true = function(df)
    for _ in range(bootstrap_samples):
      sampled_df = df.sample(n=len(df), replace=True)
      _outcome = function(sampled_df)
      treated_outcome.append(_outcome[0])
      control_outcome.append(_outcome[1])
    samples = pd.DataFrame.from_dict({"treated_outcome": treated_outcome, "control_outcome": control_outcome})
    if values:
        return samples
    else:
        #cis = samples.quantile([lower_confidence,upper_confidence])
        #lower_ci = cis[lower_confidence]
        #expected = samples.mean()
        #upper_ci = cis[upper_confidence]
        return samples#, true, expected, lower_ci, upper_ci
