# Support Vector Regression

We fed the structural features into a support vector regression model.

To avoid overfitting, we applied regularization to the regression model and estimated the out-of-sample performance on enantiomeric pairs that the model was not trained on.

The results for strictly using Morgan features were as follows:

```{figure} /modelresults/mordred.png
Caption for test
```

:::{figure-md} markdown-fig
<img src="/modelresults/mordred.png">

:::

<!-- ![morgan](../modelresults/mordred.png) -->

With a correlation metric value of 0.5:

<!-- ![alt text](https://lh3.googleusercontent.com/keep-bbsk/AP6BvTR8Pvzt9MzJYZTXMNzbyUZQah4-6sjUnw1xfVyn0GYwU9Aw-d1avuwwpR4Q-kYkIa4sMcVkKex1RmKsWdjW3VTPE1xIMeEz19flnoiXZQg0OXWD=s512) -->

The results for strictly using Mordred features were as follows:

<!-- ![alt text](https://lh3.googleusercontent.com/keep-bbsk/AP6BvTQEO11n9SZ83cHq4VbN0hcyNCiS7z08rJ_jF9yUqqDa6uYq5i52fDZn4yjVTWtqPFSneJkeS2iaXk6dWiwdfTf-zCM4WRdLx8injex0uARc1Q80=s512) -->

With a correlation metric value of 0.57:

<!-- ![alt text](https://lh3.googleusercontent.com/keep-bbsk/AP6BvTQ4cffLkIujS8Zd8_FJyYcCdpHIHmoa2QlVvaFn9xI0qQ3MocnNwxvsWkGvndDCjeXZQTfqH-kjrYA-OhgomPsAYeMx-3CKFFItQZiL70kmZQME=s512) -->
