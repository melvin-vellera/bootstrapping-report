# Bootstrapping
Lucas De Oliveira, Kooha Kwon, Melvin Vellera

## Introduction

Bootstrapping is a technique of iterative resampling a set of data with replacement used to make inferences about the target population. It is often used on small samples or when the statistic we are trying to measure does not have good parametric methods for inference (like a t-distribution).

We can bootstrap a sample by performing the following algorithm. Given a result of sample size n:

1. Take a new sample (usually of size n) from the original data with replacement.
2. Calculate the desired statistic on new sample (mean, median, standard deviation, etc.).
3. Store the statistic for later.
4. Repeat steps 1-3 many times (usually 1,000 or 10,000 times).

Lastly, analyze the stored values for inference. This could range from plotting a histogram of sample means to using confidence intervals for hypothesis tests on population parameters.

### Example: Small Result Sample from a Clinical Trial

Suppose we ran a clinical trial on a new drug to see if it had a significant effect in ameliorating symptoms of a particular disease. Our sample is small as we could only afford to conduct the trial on 8 people at this stage. We observe the following:

<p align="center">
<img src="https://github.com/lbdeoliveira/msds610_code_presentation/blob/master/assets/images/sample.png" width="800">
</p>

As we can see, some observations showed great improvement while others regressed. Of course, we don't really know the reason for the variation in the result set from such a small sample. The people that improved may have improved regardless of taking the drug (or even despite taking the drug), and similarly those that got worse might have gotten worse regardless. In order to properly infer what the range of reactions to this drug might be in the general population, we would need to repeat this experiment many more times or with a very large sample that could confer some real statistical power. Doing so, however, would likely cost many millions of dollars.

Bootstrapping this initial result set gives us a way of simulating the outcomes of repeated trials without actually having to perform an expensive experiment again (at least not yet). It also allows us to capture a larger range of estimates for the population parameter due to sampling with replacement. Note how much the sample mean (red bar) moves around in the bootstrapped samples below:

<p align="center">
<img src="https://github.com/lbdeoliveira/msds610_code_presentation/blob/master/assets/images/bootstraps.gif" width="800">
</p>

Finally, we can plot a histogram of 10,000 bootstrapped sample means and analyze their distribution:

<p align="center">
<img src="https://github.com/lbdeoliveira/msds610_code_presentation/blob/master/assets/images/samplemeans_hist.png" width="800">
</p>

Now that we have the basics down, let's do some examples with real data.



## Traditional vs. Boostrap Estimations

### Estimation of Population Mean Using Traditional and Bootstrap Methods

We'll use a real dataset to compare the traditional and bootstrap statistical estimations. Here, we have 252 rows of data of bodyfat percentage. We are going to treat this dataset as a population. The population mean of the data is 18.94%.

However, we are going to assume we only have 10 data points and build a 95% Confidence Interval with two methods. The data points were randomly sampled from the population using the following code:

```
samp_n = 10
sample_bfp = bfp.loc[random.sample(range(252), samp_n)]
sample_bfp['BODYFAT']
```

Traditionally, we can use Student's t-statistics in order to calculate the interval. The methodology requires estimation of standard deviation, standard error, and critical t*. The following equations can be used to calculate the 95% CI of the population mean.

<p align="center">
<img src="https://github.com/lbdeoliveira/msds610_code_presentation/blob/master/assets/images/CI_Equations.jpg" width="800">
</p>

Now, let's use bootstrapping to estimate. As stated before, we first need to resample the sample multiple times to simulate sample distribution. And then, we calculate the mean for each simulation and collect all the means into a list. 

```
resample_means = []

for i in range(10_000):                              # Repeating Simulation 10,000 Times
    inds = np.random.randint(0, samp_n-1, samp_n-1)  # Randomly Pulling Samples
    mean = sample_bfp['BODYFAT'].iloc[inds].mean()   # Calculating Simulated Mean
    resample_means.append(mean)                      # Collecting Means into a List
```

From the list of simulated means, we could simply locate 2.5th and 97.5th percentile values, and that will be the lower and upper bounds for the confidence interval of the population mean. We plot this confidence interval in the histogram below:

<p align="center">
<img src="https://github.com/lbdeoliveira/msds610_code_presentation/blob/master/assets/images/Bootstrap_Mean_Histogram.jpg" width="800">
</p>

The table below shows the true population distribution, mean estimation using the traditional method, and estimation using the bootstrapping method. Both traditional and bootstrapping methods were able to successfully include the true population mean within their 95% CI. However, the bootstrapping method outperforms the traditional method with a tighter interval range.

<p align="center">
<img src="https://github.com/lbdeoliveira/msds610_code_presentation/blob/master/assets/images/Mean_Results.jpg" width="700">
</p>

Next, let's consider a case where we cannot use a parametric estimation to make inferences about a population statistic.


### What if you do not know the analytical solution for calculating the confidence interval of a statistic?

Suppose we run the following regression: *BODYFAT ~ ABDOMEN + WRIST + ANKLE + AGE*

We want to know what the Adjusted R2 would be for this relationship across the general population. (For the sake of simplicity we will refer to Adjusted R2 as R2 from now.)

We first calculate R2 for the entire dataset and assume that this is the population R2 that we need to estimate from a sample.

```
model = smf.ols('BODYFAT ~ ABDOMEN + WRIST + ANKLE + AGE', data=bfp).fit()
pop_adj_rsquared = model.rsquared_adj
print(f'Population Adjusted R2: {pop_adj_rsquared:.2f}')
```
> **Population Adjusted R2: 0.71**

We now take a random sample of 30 rows from the entired dataset and assume that we do not have the rest of the data.

```
samp_n = 30
sample_bfp = bfp.loc[random.sample(range(pop_n), samp_n)]
```

Let's see what the sample R2 looks like...

```
sample_model = smf.ols('BODYFAT ~ ABDOMEN + WRIST + ANKLE + AGE',
                       data=sample_bfp).fit()
samp_adj_rsquared = sample_model.rsquared_adj
print(f'Sample Adjusted R2: {samp_adj_rsquared:.2f}')
```
> **Sample Adjusted R2: 0.75**

As you see, the sample R2 might not match the population R2. It could also be wildly different if our sample is not representative of the population. So for bootstrapping to work, it is essential for the sample to be representative of the population. 
We know, by design, that our sample is representative of the population.
Now let's calculate a bootstrapped confidence interval for R2 from our sample.

```
adj_rsquareds = []
for i in range(1_000):                                            # Iterate a 1000 times 
    inds = np.random.randint(0, samp_n-1, samp_n-1)               # Random sampling with replacement 
    model = smf.ols('BODYFAT ~ ABDOMEN + WRIST + ANKLE + AGE',    # Fit model
                    data=sample_bfp.iloc[inds]).fit()             
    rsq_adj = model.rsquared_adj                                  # Get R2
    adj_rsquareds.append(rsq_adj)                                 # Store R2

boot_rsq_adj_mean = np.mean(adj_rsquareds)                        # Calculate mean of 1000 R2 values
boot_rsq_adj_ci = [np.quantile(adj_rsquareds, 0.025),             # Calculate 2.5th and 97.5th percentiles (R2)
                   np.quantile(adj_rsquareds, 0.975)]

print(f'Bootstrap Confidence Interval (Adj Rsquared): '
      + f'''{[f'{e:.2f}' for e in boot_rsq_adj_ci]}''')
print(f'Population Rsquared_adj: {pop_adj_rsquared:.2f}')
```
>**Bootstrap Confidence Interval (Adjusted R2): ['0.57', '0.93']<br>
>Population Adjusted R2: 0.71**

**As seen from the above output, the population R2 is indeed contained in the 95% confidence interval created from our sample!**

Let's see the histogram for our 1000 R2 values:

```
fig, ax = plt.subplots(figsize=(10, 8))

ax.hist(adj_rsquareds, alpha=.5, edgecolor='grey')
ax.vlines(x=boot_rsq_adj_mean, ymin=0, ymax=300, color='red')
ax.vlines(x=boot_rsq_adj_ci[0], ymin=0, ymax=80, color='red')
ax.vlines(x=boot_rsq_adj_ci[1], ymin=0, ymax=80, color='red')
ax.vlines(x=boot_rsq_adj_mean, ymin=0, ymax=280, color='red')

ax.annotate(f'{boot_rsq_adj_mean:.2f} (Mean)', (boot_rsq_adj_mean-0.007, 305))
ax.annotate(f'{boot_rsq_adj_ci[0]:.2f} (2.5th)',
            (boot_rsq_adj_ci[0]-0.007, 85))
ax.annotate(f'{boot_rsq_adj_ci[1]:.2f} (97.5th)',
            (boot_rsq_adj_ci[1]-0.007, 85))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_title('Distribution of Bootstrapped Sample Adj Rsquareds', size=16)

plt.show()
```
![R2 Histogram](https://github.com/lbdeoliveira/msds610_code_presentation/blob/master/assets/images/R2_Histogram.png)

From the above output, we can see that boostrapping can work even if the histogram (sampling distribution) is not a symmetric normal distribution. In fact, it does not even have to be normal! You just need the percentile values to calculate confidence intervals or to perform hypothesis testing.

### Summary
1. Bootstrapping work wells for **very small sample sizes**.
2. For such small sample sizes, it can give more **precise confidence intervals** (i.e. smaller intervals) as compared to standard statistical methods.
3. Boostrapping can be used for almost ANY statistic! (Even for ones that do not have a **normally distributed** sampling distribution, or for ones for which statistical calculations have **not been discovered** yet!)

To conclude, when everything else fails, you can always pull yourself up by your bootstraps and start bootstrapping to get some confidence (intervals).

**Thank you for reading!**
