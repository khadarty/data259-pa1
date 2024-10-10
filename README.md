[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zBaegVLR)
# Programming Assignment 1
**DATA 259**  
**Fall 2024**

## Introduction
For questions on this and other assignments, you may need to write Python code to do data analysis, provide only an explanation in prose, or do both. In this course, we will be using Quarto documents to author assignment submissions and reports to develop our skills in reproducible research tools. Use a code chunk for any code you write in solving problems, and use Markdown for your explanation of what you did. We will not count your answer as correct if we cannot see it/it is written in a code comment.

We recommend one code chunk for each question, but feel free to add more as you see fit. In general, within the Markdown, you should explain the approach you took in your code at a high level in addition to explicitly answering any questions that were asked. It is up to you to decide what code-based analyses, if any, are appropriate for a particular problem.

When you are finished, please render your document to a PDF and upload your assignment in Gradescope. Make sure to select the areas of the page corresponding to the questions on the assignment outline. It is much easier for the graders to give you feedback this way, and you will therefore get your homework assignments back faster. If there is a lot of excess output, either revisit your code to make sure you are not printing excessively, or delete the pages with excess output from the PDF before submitting.

---

## Problem 1
A financial auditor is assessing the risk that different individual institutions across the country have taken while giving grants to people. The auditor uses an opaque, inaccessible machine learning model. They have applied this model to 10,000 records of loan data that represent loans made to adults in the US.

It is unclear to us how the ML model works. All we know is that the model takes as input a row of the table and produces a label (`ml_risky_pred`) whose possible values are 1, 0, or -1.

- A value of **1** indicates that a loan was given to a person classified as ‘risky’. A person classified as ‘risky’ is one who is not likely going to repay the loan according to the ML algorithm.
- A value of **0** indicates that the loan is not risky and that the person who received it is likely to repay the loan.
- In addition, in some cases, the model produces a **-1**, which is to be interpreted as ‘prediction unavailable’.

From the perspective of a financial institution being assessed by this auditor, it is in their interest to have few loans that were granted to ‘risky’ people. That indicates that the companies’ own risk models are working well. From the perspective of a state, it is in their interest that few financial institutions in their territory have granted few loans to ‘risky’ people. Doing so promotes economic stability.

In `loans_data.csv`, you have all loan data available, along with the label provided by the black-box ML model. There is accompanying documentation (`loans_data.md`) explaining what each attribute means.

```python
import pandas as pd

loan_data = "loans_data.csv"
df = pd.read_csv(loan_data)
```

1. Suppose the dataset contains all relevant loans, i.e., the entire population of loans in the US. What is the percentage of current loans granted in Illinois that are risky? How does that compare with the whole US? If you find missing values, you can report them separately and you do not need to deal with them in this case. (Note: Current is one of the categories of the attribute `loan_status`; check the documentation for more info).

2. Suppose the dataset does not represent all relevant loans, but instead only a sample of them. Repeat the previous question, but now knowing that the loans are a sample of the total population of loans.

3. Regardless of reality, your boss wants to make the claim that the percentage of current risky loans in the state of Illinois is much lower than those deemed risky in the entire US. They want to take advantage of the situation to claim that financial institutions in the state of Illinois have better risk management than other states’ institutions. Regardless of whether this claim is accurate and justifiable, the analyst produced the code you see below to make the point. Take a look at the code the analyst wrote:

```python
population = df[(df['state'] == 'IL') & (df['loan_status'] == 'Current')]['ml_risky_pred']

sample_size = 50
for i in range(500000):
    sample = population.sample(sample_size, replace = False)
    num_risky_loans = sum(sample)
    if num_risky_loans <= 10:
        print("Found it: ", str(num_risky_loans))
        break

sample_proportion = num_risky_loans / len(sample)
sample_proportion
```
0.56

```python
# random sample, so independence condition holds
# testing success-failure condition (using the null value)
mean = 0.5
condition_one = sample_size * mean
condition_two = sample_size * (1 - mean)
condition_one > 10 and condition_two > 10
```
True

```python
import math
# standard error of normal distribution for the p-value
se = math.sqrt((mean*(1 - mean)) / sample_size)
se
```
0.07071067811865475

```python
# If the null hypothesis is true, then the null distribution follows a normal with:
# mean = 0.5
# se = 0.07
# Our point estimate is 0.2. We now find the tail area for that one
z = (sample_proportion - mean) / se
z
```
0.8485281374238578

```python
import scipy.stats
# find p-value for two-tailed z-test
p_value = scipy.stats.norm.sf(abs(z))*2
p_value
```
0.39614390915207376

```python
p_value < 0.05
```
False

4. The analyst claims they can reject the null hypothesis because p < 0.05. Furthermore, they defend they used a random sample so the independence condition holds. They use this evidence to claim that the percentage of risky loans in Illinois is lower than elsewhere. Having studied the code, provide a list of bullet points with the problems you see and that you would communicate to a decision maker to persuade them not to use this analysis.

5. What if you did not have access to the code or the data? Instead, the analyst tells you that they found significant evidence that the number of risky loans in Illinois is below the national average. What would you have to do to test that claim?

---

## Problem 2
An analyst has run a number of experiments and obtained the following p-values for each. Correct for multiple comparisons using **Bonferroni methods** and **False Discovery Rate**. Assume alpha = 10%. *In this case, we ask you to write the code yourself, as opposed to using an existing library.*

```python
p_values = [0.004, 0.44501577, 0.74140679, 0.0003, 0.0040743296, 0.40743933, 
            0.94285637, 0.00158846, 0.31936529, 0.70628362, 0.3215325, 
            0.32070448, 0.5955953, 0.02785609, 0.04227114, 0.28696007, 
            0.0057042, 0.6233334, 0.18193275, 0.07893028, 0.00928628, 
            0.41068771, 0.5269194, 0.077115, 0.00308907, 0.54416113, 
            0.12486744, 0.64642929, 0.27404033, 0.38526039, 0.27368472, 
            0.96800706, 0.49461555, 0.14509363, 0.0461658, 0.0007261, 
            0.58272264, 0.02501718, 0.09205833, 0.57803194, 0.76988452, 
            0.5680329, 0.45396565, 0.38166771, 0.06963406, 0.23581046, 
            0.3225289, 0.8547721, 0.63443332, 0.03894686, 0.62706277, 
            0.35008823, 0.24922772, 0.72962402, 0.84872948, 0.82414566, 
            0.20067363, 0.37857999, 0.62977724, 0.0005504, 0.31590734, 
            0.4263561, 0.0009078, 0.00180797, 0.79175501, 0.9124886, 
            0.47129693, 0.84219809, 0.64118798, 0.25942479, 0.00109813, 
            0.93798016, 0.48571054, 0.94116676, 0.00439978, 0.79443381, 
            0.53468295, 0.38246722, 0.53655595, 0.02342969, 0.41306335, 
            0.63949623, 0.003028193, 0.30213487, 0.20940324, 0.30791922, 
            0.82000972, 0.62882809, 0.0021391, 0.69611787, 0.005386676, 
            0.83363883, 0.24132303, 0.37158356, 0.34748915, 0.07166326, 
            0.61643089, 0.00097506, 0.00103997, 0.4072646]
```

1. If alpha = 10%, how many experiments would correspond to a discovery (without correction)?

2. Write code to correct for multiple comparisons using **Bonferroni correction**. How many discoveries you made after the correction and what are the corresponding p-values?

3. Look up the **Holm-Bonferroni** correction procedure. Then, write code to correct for multiple comparisons using Holm-Bonferroni correction. How many discoveries you made after the correction and what are the corresponding p-values?

4. Correct for multiple comparisons using **False Discovery Rate**. How many discoveries you made after the correction and what are the corresponding p-values? Assume the maximum false discovery rate 𝑄 to be the same as alpha.

5. Comment on your observations. How do these three methods compare with each other?
