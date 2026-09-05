# Generative Modelling Overview

This is a repository for me to try out different generative modelling architectures. 

## My current interests: 
* GANs 
* Diffusion Models 
* Transformer Decoders
* Autoregressive Models
* Autoencoders
* Graph Neural Networks (GNNs)

### An overview to return to of Supervised Learning

**1. We assume there exists a **true, unknown** conditional distribution:**

$p_{true}(Y | X = x)$

Keep in mind that Y is a random variable. A random variable is formally defined to be a function: 

$Y : \Omega \rightarrow \mathbb{R}$

Where $\Omega$ is the set of outcomes.

So for a less brief notation, we assume there exists a **true, unknown** conditional distribution:

$p_{true}(Y(\omega) = y \: | \: X(\omega) = x)$

**2. We want to approximate the true distribution with our own parameterized distribution.**

$p_{\theta} = p_{true}$

**3. In reality, we don't have the true distribution of our data, we only have labelled data points that represent one joint realization of N labeled observations.**

$D = {(x_i, y_i)}_{i=1}^N$

At this point, we move from the land of probability (random variables, true distributions) to the land of statistics. Here, we used real life observed values to approximate probabilistic distributions. 

**4. Now, we assume that our data is IID. This allows us to use probabilistic independence. We say that:**

$L(\theta; D) = p_{\theta}(y_1, ..., y_N | x_1, ..., x_N) =  p(y_1 | x_1) * p(y_2 | x_2) \: ... * p(y_N | x_N)$

Keep in mind that at this point, the data set is fixed and $L(\theta; D)$ evaluates to an actual number. 

Rewrite the above expression into compact notation: 

$L(\theta; D) = \prod_{i=1}^N p_{\theta}(y_i | x_i)$

Once again, the fully explicit notation is given as such:

$L(\theta; D) = \prod_{i=1}^N p_{\theta}(Y(\omega) = y_i |X(\omega) = x_i)$

Here it is clear that for given parameter set $\theta$ we will retrive a number. That number is the likelihood of our dataset, which we want to maximize. Our goal here is vary $\theta$ in order to maximize the likelihood of our data. This should leave us with a parameter set $\theta$ such that $p_{\theta} \approx p_{true}$

**5. The inferential leap**

At this point, we should emphasize that

$p_{\theta}(y | x)$ is a function defined over all possible $(x,y)$, 
while $p_{\theta}(y_i | x_i)$ is that same function evaluated at one single point. 

MLE only constrains $\theta$ through the finite set of observed evaluations, but because the same parameter vector $\theta$ determines the function everywhere, fitting those points also determines its values at unobserved points. 

**Here is a key point of confusion**

In the MLE formulation, we have $L(\theta; D) = \prod_{i=1}^N p_{\theta}(Y(\omega) = y_i |X(\omega) = x_i)$

However, we then assume that the best $\theta$ retrieved from maximizing the likelihood of the data is suitable as the true function $p_{true}$. This is a very bold claim. We are assuming that a function $p_{\theta}$ that is rigid with a finite number of parameters can approximate a true probability function. This in reality is never the case, we cannot perfectly model the true distribution under this formulation. 

Additionally, I will add that the reason we can use observations $(y_i,x_i) to then generalize to all $(x,y)$ is precisely because we induce a bias on the proposed function $p_{\theta}$. $\theta$ is specifically a rigid parameter set that takes on values, this is simply a function that can take any input because we have made it rigid with actual values. This allows us to induce bias that lets us model all inputs. However, it comes at the cost of the true flexibility of the $p_{true}$ function. 


6. The MLE formulation

Our goal is to find the $\theta$ to maximize the likelihood of our data.

$\hat\theta_{MLE} =  arg max_{\theta}(L(\theta; D)) = arg max_{\theta} \prod_{i=1}^N p_{\theta}(Y(\omega) = y_i |X(\omega) = x_i)$

From here, we turn to logs for numerical stability:

$\hat\theta_{MLE} = arg max_{\theta} \sum_{i=1}^N log(p_{\theta}(Y(\omega) = y_i |X(\omega) = x_i))$

7. We assume the model family for Y | X = x
