# Hypothesis Testing for Better Decisions

Statistics is not about math. It is about turning real world problems into standardized tests so you can make better decisions. Hypothesis testing is the tool that lets you say something is statistically significant.

## Why Standardization Matters

You do not want one test for ping pong and another test for football and another test for hot dog eating. You want one approach that works across all domains.

Standardization takes real life and turns it into a statistics problem. You divide everything by the mean. This moves you from the real world with actual values to a standardized world where the mean is zero.

As numbers increase, you talk about being one standard deviation away from the mean or two standard deviations away. This lets you compare different activities using the same framework.

The statistics are not valuable by themselves. They only matter because they tell you something about the world. You take a statistical test and turn it back into real world language that lets you make better decisions.

## The Bell Curve and Percentages

With a bell curve, the area under the curve adds up to one. This is handy because you can turn it into a percentage. If you divide it in half, 50% of your values are on one side and 50% are on the other side.

But you usually do not care about 50%. You care about capturing 95% of the values. This is where you talk about the orange area plus the red areas on a distribution. That gives you close to 95% of all the values that would fall in that space.

This is not arbitrary. 95% is the standard everyone uses. It means you are willing to be wrong 5% of the time. That 5% is called alpha. Most of the time you split alpha between the high side and the low side, so you have 2.5% on each tail.

## The Null and Alternative Hypothesis

The null hypothesis is the status quo. It is the way things are. It is written as H0 or H sub zero. The alternative hypothesis says no, something else is true. It is written as HA or H1.

The null hypothesis always has an equals sign in it. The alternative hypothesis cannot have an equals sign. This is how you tell them apart.

What you normally care about from a business sense is the alternative hypothesis. You want to show that something is different. You want to show that a change had an effect. You want to show that one group performs better than another.

The null hypothesis is already accepted. It can only be rejected. The burden of proof is on the alternative. This is the structure you work within.

## The Null Hypothesis Poem

I am what is the default, the status quo. I am already accepted, can only be rejected. The burden of proof is on the alternative. I am the null hypothesis.

This poem captures the logic. You start by assuming nothing has changed. You start by assuming there is no difference. Then you look at your data and ask if it is weird enough to reject that assumption.

## The Steps of Hypothesis Testing

First, define your hypothesis. Decide what your null hypothesis is and what your alternative hypothesis is.

Second, decide on your critical value. This is your test statistic. You will use Excel or you will remember that 1.96 is the value for a 95% confidence level with a two-tailed test.

Third, define your rejection region. This is the area where if your test statistic falls there, you will reject the null hypothesis.

Fourth, do the test. Calculate your test statistic from your data.

Fifth, make a conclusion. You will either reject the null hypothesis or fail to reject the null hypothesis. You cannot accept the null hypothesis. It is already accepted.

## Two-Tailed vs One-Tailed Tests

A two-tailed test is when you care about being too high or too low. You do not know which direction the difference will be. You just want to know if there is a difference.

A one-tailed test is when you only care about one direction. Maybe you only care if something is higher than a threshold. Maybe you only care if something is lower than a threshold.

In general, use two-tailed tests. This is what everyone does. This is the standard. Use a one-tailed test only when you have a specific reason to care about just one direction.

## The Rejection Region

The rejection region is where your test statistic has to fall for you to reject the null hypothesis. If you do a test and your Z stat or T stat is in the rejection region, that means the data you observed would be weird if the null hypothesis were true.

For example, if the null hypothesis says the true mean is 35, but your data says the mean is 18, you ask: is that likely to happen if the true mean is 35? If the answer is no, you reject the null hypothesis.

The rejection region is defined by your critical value. For a 95% confidence level with a two-tailed test, the critical value is 1.96. If your test statistic is more than 1.96 standard deviations away from the mean, you reject the null hypothesis.

## Confidence Intervals and Hypothesis Tests

Confidence intervals and hypothesis tests are friends. They work together. A confidence interval gives you a range of plausible values for the true parameter. A hypothesis test tells you whether a specific value is plausible.

If someone asks if the true population parameter is 9, and 9 does not fall within your confidence interval, the answer is no. If someone asks if the true population parameter is 13, and 13 does fall within your confidence interval, the answer is maybe.

You can also use confidence intervals to compare two groups. If the confidence intervals overlap, you do not have evidence that the groups are different. If the confidence intervals do not overlap, you have evidence that the groups are different.

For example, North Dakota and South Dakota both had commute times around 17 minutes. The confidence intervals overlapped. You cannot say the commute times are different. But New York state had a commute time of 33 minutes. The confidence intervals did not overlap. You can say the commute time is different.

## Bootstrapping as a Sensitivity Check

Bootstrapping is a way to simulate what would happen if you had collected different data. You take your sample and systematically remove one person, calculate the mean, put that person back, remove someone else, calculate the mean again.

You do this many times. This creates a distribution of possible means. You can use this distribution to create a confidence interval. You can use it to see how sensitive your results are to individual data points.

Bootstrapping does not require you to assume your data follows a normal distribution. It uses the data you have to estimate the variability. This makes it useful when you are not sure about the underlying distribution.

## When to Use T vs Z

If you have less than 30 observations, use the t-test. If you have 30 or more observations, use the z-test.

The t-distribution has fatter tails than the normal distribution. This accounts for the extra uncertainty when you have a small sample. As your sample size increases, the t-distribution gets closer to the normal distribution.

You do not need to calculate these values by hand. Use Excel. For the t-value, use T.INV with your alpha level and degrees of freedom. For the z-value, use NORM.S.INV with 0.975 for a 95% two-tailed test.

## The Phrase That Matters

After you do a hypothesis test, you get to use the phrase statistically significant. This phrase means you did a test and the results are valuable. The results are not likely to be due to chance.

Anytime you hear in the news that something is statistically significant, it means they should have done one of these tests. It means they have evidence that the effect is real.

This phrase carries weight. Use it correctly. Do the test. Follow the steps. Make the conclusion. Then you can say the results are statistically significant.

## The Real World Connection

Statistics is a tool. The tool only matters if it helps you make better decisions in the real world. You take a business problem. You turn it into a statistical problem. You solve the statistical problem. Then you turn the answer back into business language.

Do not get lost in the math. The math is a means to an end. The end is a better decision. The end is knowing whether a change worked. The end is knowing whether two groups are different. The end is knowing whether to act.

Hypothesis testing gives you a structured way to answer these questions. It gives you a way to quantify uncertainty. It gives you a way to say with confidence that something is true or not true.

## How to Use This

When you have a business question, frame it as a hypothesis. What is the null hypothesis? What is the alternative hypothesis?

Collect data. Calculate your test statistic. Compare it to the critical value. Make a conclusion.

If you reject the null hypothesis, you have evidence for the alternative. You can say the effect is statistically significant. You can act with confidence.

If you fail to reject the null hypothesis, you do not have evidence for the alternative. You cannot say the effect is real. You should not act as if it is.

This structure keeps you honest. It keeps you from seeing patterns that are not there. It keeps you from making decisions based on noise.

## The Bottom Line

Hypothesis testing is a tool for making better decisions under uncertainty. You standardize the problem. You define the null and alternative hypothesis. You collect data. You calculate a test statistic. You compare it to a critical value. You make a conclusion.

If the test statistic falls in the rejection region, you reject the null hypothesis. You have evidence for the alternative. You can say the result is statistically significant.

If the test statistic does not fall in the rejection region, you fail to reject the null hypothesis. You do not have evidence for the alternative. You cannot say the result is significant.

This framework works across domains. It works for ping pong and football and hot dog eating. It works for business decisions. It works whenever you need to know if an effect is real or just noise.

Use it. Follow the steps. Make better decisions.
