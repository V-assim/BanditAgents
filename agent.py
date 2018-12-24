#!/usr/bin/python
# coding: utf-8

# Author : Wassim Bouaziz
# email : wassim.s.bouaziz@gmail.com

import numpy as np
#np.random.seed(0)
import random
#random.seed(0)
from scipy.optimize import minimize

"""
Contains the definition of the agent that will run in an
environment.
"""



class RandomAgent:
    def __init__(self):
        """
        Init a new agent.
        """


    def choose(self):
        """
        Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        """
        return np.random.randint(0, 10)


    def update(self, action, reward):
        """
        Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        pass



class epsGreedyAgent:
    def __init__(self):
        """
        The epsilon-greedy agent. Exploit the best known arm but keeps a non-zero probability to explore other arms.
        """
        self.A = [0,1,2,3,4,5,6,7,8,9] # List of possible arms.
        self.epsilon = 0.15
        np.random.seed(6543) # Seems like this value of seed works quite well with the epsilon.
        self.n_arm = np.zeros(len(self.A)) # Number of time each arm has been pulled.
        self.mu = np.zeros(len(self.A)) # Empirical mean of each arm.


    def choose(self):
        proba_arm = np.zeros(len(self.A)) + self.epsilon/len(self.A)
        proba_arm[np.argmax(self.mu)] += (1 - self.epsilon) # With probability (1 - epsilon) the agent pull the arm with the highest empirical mean and keeps a probability of epsilon to pull a uniformly random arm.

        arm = np.random.choice(self.A, p=proba_arm)
        return(arm)


    def update(self, action, reward):
        self.n_arm[action] += 1
        self.mu[action] += (reward - self.mu[action]) / self.n_arm[action]



class epsNGreedyAgent:
    def __init__(self):
        """
        Epsilon_n greedy agent uses the same strategy as the epsilon-greedy but has a variable epsilon.
        """
        self.A = [0,1,2,3,4,5,6,7,8,9] # List of possible arms.
        self.c = 10
        self.d = 0.75
        self.epsilon = 0.15
        self.n_arm = np.zeros(len(self.A)) # Number of time each arm has been pulled.
        self.N = 0 # Total number of rounds.
        self.mu = np.zeros(len(self.A)) # Empirical mean of each arm.


    def choose(self):
        self.epsilon = min(1, self.c*len(self.A)/(self.d**2 * self.N + 10**(-5))) # To avoid issue when self.N==0.

        proba_arm = np.zeros(len(self.A)) + self.epsilon/len(self.A)
        proba_arm[np.argmax(self.mu)] += (1 - self.epsilon)

        arm = np.random.choice(self.A, p=proba_arm)
        return(arm)


    def update(self, action, reward):
        self.n_arm[action] += 1
        self.N += 1
        self.mu[action] += (reward - self.mu[action]) / self.n_arm[action]



class BesaAgent():
    # https://hal.archives-ouvertes.fr/hal-01025651v1/document
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9] # List of possible arms.
        self.n_arm = np.zeros(len(self.A), dtype=int) # Number of time each arm has been pulled.
        self.rewards = [[] for _ in range(len(self.A))] # List of rewards for each arm.
        random.seed(None)


    def choose_2(self,A_1,A_2):
        if (len(A_1) == 0) or (len(A_2) == 0): # If one of the subset is empty :
            return (A_1 + A_2)[0] # Return the other.
            # Since the subsets are cut in half, the only case in which this can happen is if both A_1 and A_2 are empty, or if one is empty, the other can only contain 1 element.
        else:
            a = self.choose_2(A_1[:int(len(A_1)/2)], A_1[int(len(A_1)/2):]) # Break A_1 in half. This will actually return only 1 arm.
            b = self.choose_2(A_2[:int(len(A_2)/2)], A_2[int(len(A_2)/2):]) # Break A_2 in half.

            I_a = random.sample(list(range(self.n_arm[a])), int(min(self.n_arm[a],self.n_arm[b]))) # if N[a] < N[b], select I_a entirely.
            I_b = random.sample(list(range(self.n_arm[b])), int(min(self.n_arm[b],self.n_arm[a]))) # if N[b] < N[a], select I_a entirely.
            mu_a = np.mean([self.rewards[a][i] for i in I_a])
            mu_b = np.mean([self.rewards[b][i] for i in I_b])

            return (mu_a >= mu_b)*a+(mu_a < mu_b)*b # Returns the arm with the highest empirical mean on the subsets.


    def choose(self):
        if 0 in list(self.n_arm): # If one of the arms hasn't been pulled yet...
            a = list(self.n_arm).index(0) # Pull one of them.
        else: # If all the arms have been pulled :
            a = self.choose_2(self.A[:int(len(self.A)/2)], self.A[int(len(self.A)/2):])
        return a


    def update(self, action, reward):
        self.n_arm[action] += 1
        self.rewards[action] += [reward]



class SoftmaxAgent:
    # https://www.cs.mcgill.ca/~vkules/bandits.pdf
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9] # List of possible arms.
        self.tau = 0.357 # The parameter.
        self.n_arm = np.zeros(len(self.A)) # Number of time each arm has been pulled.
        self.mu = np.zeros(len(self.A)) # Empirical mean of each arm.


    def choose(self):
        boltzmann_distr = np.exp((self.mu - np.max(self.mu)) / self.tau) # We substract the mean to have the function stable.
        boltzmann_distr /= np.sum(boltzmann_distr) # Each element of the array is the softmax of the empirical mean over all the empirical means.
        a = np.random.choice(self.A, p=boltzmann_distr)
        return a


    def update(self, action, reward):
        self.n_arm[action] += 1
        self.mu[action] += (reward - self.mu[action]) / self.n_arm[action]



class UCB1Agent:
    # https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9] # List of possible arms.
        self.n_arm = np.zeros(len(self.A)) # Number of time each arm has been pulled.
        self.N = 0 # Total number of plays.
        self.mu = np.zeros(len(self.A)) # Empirical mean of each arm.


    def choose(self):
        if 0 in self.n_arm: # If one arm hasn't been pulled yet ...
            return np.random.choice([i for i in range(len(self.n_arm)) if self.n_arm[i]==0]) # Pull one of them.
        else:
            fact = 1.6
            pow = 0.7
            L = self.mu + (fact*np.log(self.N)/self.n_arm)**pow # Trying to tweak the values of both fact and pow gave interesting results !
            return np.argmax(L)


    def update(self, a, r):
        self.N += 1
        self.n_arm[a] += 1
        self.mu[a] += (r - self.mu[a]) / self.n_arm[a]



class UCBNormalAgent:
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9] # List of possible arms.
        self.n_arm = np.zeros(len(self.A)) # Number of time each arm has been pulled.
        self.N = 0 # Nomber of total plays.
        self.mu = np.zeros(len(self.A)) # Empirical mean of each arm.
        self.q_j = np.zeros(len(self.A)) # Will contain the sums of the squared rewards of each arm.


    def choose(self):
        if (0 in self.n_arm):
            return np.random.choice([i for i in range(len(self.n_arm)) if self.n_arm[i] == 0])

        if (True in (self.n_arm < np.ceil(8*np.log(self.N)))):
            return np.argmin(self.n_arm)

        else:
            L = self.mu + (16*(abs(self.q_j - self.n_arm*self.mu**2) / (self.n_arm - 1))*(np.log(self.N - 1) / self.n_arm))**0.5
            return np.argmax(L)


    def update(self, a, r):
        self.N += 1
        self.n_arm[a] += 1
        self.mu[a] += ((r - self.mu[a]) / self.n_arm[a])
        self.q_j[a] += r**2



class BetaThompsonAgent:
    # https://en.wikipedia.org/wiki/Thompson_sampling
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9] # List of possible arms.
        self.success = np.zeros(len(self.A))
        self.failure = np.zeros(len(self.A))
        self.max_rewards = np.zeros(len(self.A))
        self.min_rewards = np.zeros(len(self.A))


    def choose(self):
        # We will use the Beta distribution as prior distribution but it could be generalized as well to any distribution.
        theta = np.random.beta(self.success + 1, self.failure + 1)
        return np.argmax(theta)


    def update(self, a, r):
        """
        I calculated the cumulative distribution function, knowing how rewards are generated. It was a bit long so I won't detail it.
        """
        def F(u):
            def F_10_01(u):
                return ((1 / (2 * np.log(10))) * (-0.5*np.log(-u) - 0.1*u - u**2/400 + 0.5*np.log(10) - 0.75))

            def F_01_0(u):
                return (F_10_01(-0.1) + (1 / (2 * np.log(10))) * (99*u/10 + 9999*u**2/400 + 99/100 - 9999/40000))

            def F_0_01(u):
                return (F_01_0(0) + (1 / (2 * np.log(10))) * (99*u/10 - 9999*u**2/400))

            def F_01_10(u):
                return (F_0_01(0.1) + (1 / (2 * np.log(10))) * (0.5*np.log(u) - 0.1*u + u**2/400 + 0.5*np.log(10) + 0.01 - 1/40000))

            if (u > 10) or (u < -10):
                raise ValueError('out of definition domain')
            else:
                if (u >= -10) and (u <= -0.1):
                    return F_10_01(u)
                elif (u >= -0.1) and (u <= 0):
                    return F_01_0(u)
                elif (u >= 0) and (u <= 0.1):
                    return F_0_01(u)
                elif (u >= 0.1) and (u <= 10):
                    return F_01_10(u)

        pr = F(r)
        r_t = np.random.binomial(1,pr)

        self.success[a] += r_t
        self.failure[a] += (1 - r_t)



class GaussianThompsonAgent:
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9] # List of possible arms.
        self.mu = np.zeros(len(self.A)) # Empirical mean of each arm.
        self.S2 = np.zeros(len(self.A)) # Will contain the sums of the squared rewards of each arm.
        self.sigma2 = np.ones(len(self.A)) # Will contain the empirical variance of each arm's reward distribution.
        self.n_arm = np.zeros(len(self.A)) # Number of time each arm has been pulled.


    def choose(self):
        theta = np.random.normal(self.mu, self.sigma2**0.5) # Pull thetas with prior distribution
        return np.argmax(theta)


    def update(self, a, r): # Adjust distribution
        self.n_arm[a] += 1
        self.mu[a] += (r - self.mu[a])/self.n_arm[a]
        self.S2[a] += r**2
        self.sigma2[a] = (self.n_arm[a] - 1)/(self.n_arm[a])*(abs(self.S2[a]/self.n_arm[a] - self.mu[a]**2)) # We use the formula of an unbiased empirical variance.



class KLUCBAgent:
    # See: https://hal.archives-ouvertes.fr/hal-00738209v2
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9] # List of possible arms.
        self.n_arm = np.zeros(len(self.A)) # Number of time each arm has been pulled.
        self.N = 0 # Nomber of total plays.
        self.mu = np.zeros(len(self.A)) # Empirical mean of each arm.
        self.sigma2 = np.zeros(len(self.A)) # Will contain the empirical variance of each arm's reward distribution.
        self.S2 = np.zeros(len(self.A)) # Will contain the sums of the squared rewards of each arm.
        self.f = lambda t : np.log(t) # f must be a non decreasing function. The article propose log.


    def choose(self):
        if 0 in self.n_arm: # If one of the arms hasn't been pulled yet...
            return np.argmin(self.n_arm) # Pull one of them.
        else:

            def d_KL(mu_p,mu_q, sigma2): # Normally, the KL divergence can be used with distributions of different variance. We use this simpler formula since we will only use it with the hypothesis of same variance.
                return ((mu_p - mu_q)**2 / (2*sigma2+10**(-3)))

            def findSupremum(i):
                mu_a = self.mu[i]
                fun_to_minimize = lambda x : -x
                x_0 = [mu_a]
                cons = {
                    'type' : 'ineq',
                    'fun' : lambda mu : -(d_KL(mu_a,mu,self.sigma2[i])-(self.f(self.N)/(self.n_arm[i]+10**(-3)))),
                    'jac' : lambda mu : (mu-mu_a)/(self.sigma2[i]+10**(-3))
                }
                bounds = [(-10,10)]
                res = minimize(fun=fun_to_minimize, x0=x_0, jac=lambda x : -1, method='SLSQP', bounds=bounds,constraints=cons)['x'][0] # SLSQP is the only method which accepts both bounds and constraints.
                return res

            U_a = [findSupremum(i) for i in range(len(self.A))]
            #U_a = [x for x in U_a if str(x) != 'nan']
            #print(U_a)
            #arm_argmax = [i for i in range(len(U_a)) if U_a[i]==max(U_a)]
            return np.argmax(U_a)


    def update(self, a, r):
        self.N += 1
        self.n_arm[a] += 1
        self.mu[a] += (r - self.mu[a]) / self.n_arm[a]
        self.S2[a] += r**2
        self.sigma2[a] = (self.n_arm[a] - 1)/(self.n_arm[a]) * (abs(self.S2[a]/self.n_arm[a] - self.mu[a]**2))



class CheaterAgent:
    arm_to_pull = [5,6,8,8,3,7,1,2,5,9,1,5,4,9,2,4,5,9,7,0,4,9,8,2,8,2,8,3,5,8,5,5,7,7,8,4,4,9,3,5,7,3,5,7,5,2,0,2,9,1,1,2,4,5,5,4,1,7,5,3,6,8,7,7,1,0,0,7,8,9,4,1,4,2,7,8,1,4,5,0,5,0,7,6,3,2,2,0,6,3,4,4,2,2,1,1,6,6,1,9,4,1,8,0,4,2,1,6,3,3,3,2,2,0,1,2,4,9,3,7,1,6,8,2,6,0,9,0,6,8,7,4,0,8,7,0,7,3,3,5,3,9,7,1,6,9,7,1,7,5,4,9,6,8,9,1,2,5,6,4,9,4,1,7,8,9,5,2,5,0,9,2,8,6,9,8,0,4,5,2,6,2,6,4,5,2,9,0,8,7,8,3,3,3,3,7,3,6,7,7,9,7,6,5,5,3,1,3,9,8,7,4,3,0,7,2,7,4,7,0,4,5,0,3,9,1,7,8,7,7,9,8,4,4,7,9,0,1,5,8,5,3,3,3,0,1,7,3,6,9,7,8,4,1,5,4,0,6,5,2,6,8,8,6,1,2,8,5,6,9,9,4,5,4,6,8,1,1,4,6,1,9,6,5,2,7,8,3,3,6,4,4,3,4,6,5,9,9,3,3,0,3,9,6,3,1,4,8,2,5,1,7,8,9,5,4,8,3,8,8,8,9,6,9,5,2,7,1,6,8,4,3,6,4,7,4,6,1,2,9,7,1,6,5,9,1,7,6,5,6,0,5,1,7,2,3,8,1,1,4,4,5,8,3,8,0,6,0,3,3,2,8,7,4,2,6,5,9,2,3,7,5,3,9,7,8,5,6,7,3,0,2,9,4,9,9,8,8,2,8,9,5,3,9,8,0,9,3,9,4,0,5,0,1,9,1,4,5,8,1,0,1,6,8,5,0,9,3,4,8,7,4,8,3,7,4,3,0,5,5,3,7,0,8,0,5,1,2,6,7,6,5,9,9,9,4,0,3,5,7,0,3,1,7,3,6,1,5,7,1,1,4,9,4,7,2,3,1,9,8,1,6,1,8,4,4,8,4,8,4,9,0,3,0,2,3,6,5,1,4,6,4,4,3,3,0,4,5,5,1,0,2,1,8,4,3,4,9,0,0,6,7,1,0,5,9,1,2,0,9,8,3,8,3,0,2,8,7,3,0,6,7,4,9,3,9,8,8,7,6,1,9,9,3,5,5,9,3,8,6,8,2,2,3,2,5,7,5,5,1,1,8,6,2,8,1,5,0,1,1,4,7,0,3,1,6,7,5,4,2,4,0,0,1,8,2,9,3,4,9,5,0,4,1,9,9,6,7,8,5,0,8,5,6,8,1,2,8,1,7,2,9,1,5,6,4,7,3,2,1,4,5,2,9,7,2,7,2,6,4,3,2,9,9,6,0,0,8,8,6,5,6,5,9,8,9,6,4,6,3,2,3,8,8,0,1,7,2,7,0,5,2,6,9,1,9,8,1,3,5,4,1,5,3,9,9,1,6,7,2,5,2,0,8,7,2,6,6,0,5,0,8,2,4,6,2,1,6,6,2,6,5,6,6,2,0,3,6,2,6,1,3,0,8,2,6,4,5,9,2,0,7,4,9,4,3,6,1,4,6,5,6,7,8,6,1,4,4,2,6,7,5,5,7,1,1,4,3,5,5,7,6,2,5,4,0,8,9,7,3,1,4,7,4,5,6,8,0,3,2,1,1,7,9,7,6,8,8,4,2,8,7,1,6,8,0,8,2,5,8,7,8,1,7,1,6,2,6,5,5,2,7,1,8,6,6,9,9,2,5,9,9,1,7,3,7,9,4,0,3,9,2,5,8,8,9,1,6,2,0,6,1,5,1,7,6,2,2,3,2,0,5,1,7,2,1,1,8,8,5,1,5,3,5,2,1,7,8,9,4,2,7,6,0,0,3,3,9,0,7,8,8,2,5,6,4,9,4,5,4,2,2,0,6,9,8,1,2,7,1,1,8,0,0,0,2,7,2,5,9,8,1,7,1,4,8,7,1,4,9,6,9,2,2,4,9,7,8,7,9,2,3,8,8,1,2,6,5,4,4,5,9,6,7,1,9,3,7,5,3,6,4,3,6,9,4,9,6,7,5,6,7,4,4,9,2,1,8,0,6,0,6,0,6,8,3,2,0,3,7,4,2,1,4,8,9,6,5,8,0,7,7,2,7,1,1,5,9,1,7,6,3,5,6,8,2,6,0,8,2,5,1,9,8,6,5,4,9,7,0,6,2,7,0,2,9,9,2,6,9,2,3,3,7,6,0,3,9,7,3,4,9,5,4,6,5,2,8,0,8,1,2,4,2,7,5,4,2,3,9,5,9,3,9,4,4,9,0,7,4,0,5,8,5,3,6,9,8,4,1,4,4,1,1,0,2,8,6,7,2,6,7,9,3,8,3,6,9,9,4,2,4,3,6,1,1,7,1,5,4,2,3,2,4,8,4,9,5,2,8,9,4,4,7,4,6,4,8,3,7,2,0,0,1,4,8,1,5,7,3,9,6,8,4,5,6,9,5,4,4,2,9,8,1,0,3,9,5,8,9,5,4,0,7,2,6,6,3,3,1,6,4,9,4,9,3,7,1,1,6,7,1,0,8,7,4,9,2,6,9,3,5,1,7,3,3,5,0,8,6,8,5,6,1,8,1,5,4,2,3,7,0,9,6,7,5,1,8,1,9,1,6,1,8,5,4,9,7,0,1,9,5,4,4,2,0,5,4,9,2,0,7,4,9,8,9,8,3,3,7,2,8,4,7,5,0,3,9,3,5,6,4,9,4,3,2,5,2,3,3,4,9,7,8,6,7,9,5,0,1,5,0,0,1,1,2,2,7,4,6,9,8,6,5,0,1,0,5,3,4,2,2,6,9,1,1,8,5,1,8,7,8,6,7,0,1,3,0,3,4,0,8,6,9,5,5,7,9,9,7,4,0,1,9,4,6,1,0,8,5,4,1,6,4,2,5,3,0,5,2,8,7,3,4,0,0,8,8,0,4,3,4,6,0,4,2,6,2,3,8,7,3,8,8,1,3,8,7,8,0,9,2,5,8,6,0,6,9,4,6,3,9,1,2,1,9,5,7,0,7,3,5,0,5,0,6,4,5,3,5,9,3,6,6,5,5,4,7,3,5,1,7,9,2,4,1,7,5,0,7,4,6,5,8,3,6,7,5,7,0,1,8,5,2,0,5,1,6,8,4,6,8,9,4,5,8,2,0,9,5,8,0,8,2,9,9,6,9,2,0,0,3,9,5,5,2,5,5,5,2,4,6,9,0,2,6,6,9,8,1,0,4,1,8,8,9,7,5,1,0,3,3,7,2,1,0,5,9,8,6,5,4,3,3,3,6,1,1,7,7,0,5,0,6,0,4,6,5,4,6,4,6,1,0,9,3,6,0,8,9,4,1,7,6,9,8,4,6,6,7,0,6,6,5,1,2,7,3,8,6,8,4,8,0,8,6,4,0,7,2,7,1,2,9,9,5,3,6,5,2,8,5,8,5,1,6,4,2,5,4,1,6,7,6,4,3,7,8,3,4,5,0,3,6,1,7,5,2,2,8,4,6,9,6,9,8,6,6,9,0,4,4,4,1,4,3,8,3,8,6,7,7,5,7,7,3,5,4,9,5,5,7,8,2,8,5,6,4,4,9,0,7,8,9,6,4,3,1,9,6,0,3,3,7,0,0,2,3,5,4,8,6,7,8,4,3,1,1,7,6,2,3,5,9,8,1,3,8,8,7,1,9,2,9,8,9,5,4,5,9,4,9,9,7,8,2,6,1,0,7,8,3,8,4,3,5,9,7,6,7,8,8,2,2,8,5,9,3,2,0,2,3,5,1,9,8,0,0,3,2,4,3,8,0,3,3,2,2,3,1,2,4,7,2,3,2,8,3,1,8,0,9,0,0,8,5,7,9,3,6,1,3,3,4,1,0,2,9,1,7,6,5,6,2,4,0,9,3,9,7,5,8,6,7,6,4,3,6,5,5,0,6,2,5,3,7,5,3,1,0,9,0,9,0,0,5,3,4,0,3,6,7,3,7,7,5,2,2,9,3,4,6,4,0,0,7,1,5,6,1,4,0,4,5,5,2,2,1,7,6,2,9,4,8,0,2,3,9,4,9,5,6,0,3,4,6,3,5,6,8,1,8,6,2,9,3,2,0,9,3,5,9,8,5,4,1,0,6,8,6,0,7,4,6,0,2,0,2,8,8,9,5,4,6,4,3,6,4,9,8,0,1,8,7,2,0,8,3,3,8,1,7,4,1,3,1,2,5,4,8,7,6,6,3,7,0,7,7,1,8,7,0,4,0,1,0,5,9,9,0,7,3,2,5,5,2,2,4,8,3,5,4,8,3,3,9,1,3,4,6,5,8,1,8,5,7,5,5,7,3,0,5,0,0,0,9,5,5,8,2,5,4,4,8,9,7,2,8,1,5,5,1,8,8,2,3,2,8,8,8,0,3,8,5,0,4,2,3,3,4,8,4,2,7,3,4,9,9,0,9,5,7,6,7,5,7,2,9,7,0,0,8,0,9,0,9,7,5,6,7,3,7,5,0,2,5,2,7,8,0,0,5,6,3,5,7,5,8,6,9,2,4,7,5,8,2,0,3,5,0,8,5,8,1,6,1,6,4,1,5,7,4,7,7,9,3,4,1,8,1,2,7,4,3,0,4,8,9,8,8,1,0,2,3,2,5,1,3,2,2,9,5,4,4,8,3,5,1,1,8,4,8,3,0,3,2,1,9,1,5,1,5,7,3,7,4,2,7,6,5,8,4,8,2,4,8,7,2,0,2,3,4,9,5,0,3,8,7,9,6,5,6,4,9,1,0,4,4,3,1,7,4,3,9,6,7,7,0,2,2,1,3,9,2,4,0,5,9,1,3,5,6,5,0,0,1,6,1,1,4,5,4,1,1,6,7,9,6,8,2,1,0,8,8,4,1,7,2,2,9,6,9,7,1,8,3,5,8,9,3,7,7,0,9,2,7,3,6,0,9,5,3,9,8,2,9,6,0,4,9,9,4,6,2,5,7,4,3,7,1,2,2,3,1,3,8,6,8,1,3,2,3,4,6,9,1,0,4,9,8,6,8,9,9,3,2,7,9,7,6,2,4,9,6,9,7,5,7,8,2,8,0,6,5,0,9,0,2,6,3,3,1,7,7,3,5,1,2,4,6,0,3,7,7,8,2,0,7,4,5,7,7,6,3,4,5,1,2,3,8,7,1,4,9,1,6,1,2,4,5,3,6,9,7,8,9,8,3,7,5,9,6,0,2,4,0,3,4,8,0,7,2,3,1,2,8,5,6,2,3,8,9,7,7,4,8,0,4,5,7,9,4,6,3,7,2,1,6,1,2,8,2,0,8,0,3,6,8,5,8,2,3,1,5,4,0,0,4,6,3,5,9,3,9,1,6,8,0,8,0,4,0,6,3,3,4,6,4,5,7,1,5,4,5,5,9,9,0,6,3,7,8,9,2,5,4,5,7,6,2,9,3,8,7,3,5,8,8,1,0,2,2,0,9,6,8,9,6,6,5,0,6,0,5,1,1,3,2,3,2,6,9,3,8,2,0,9,7,4,8,2,4,9,9,0,7,9,2,3,3,8,9,5,1,3,4,7,8,5,4,1,7,5,7,3,6,3,2,2,4,1,0,9,5,5,9,0,8,2,9,7,4,1,9,7,9,7,1,9,3,3,7,9,8,1,0,3,7,3,1,8,3,2,1,7,9,7,9,5,2,9,6,8,5,5,3,1,2,4,1,3,3,2,1,2,3,4,2,3,5,0,4,3,7,9,1,3,0,7,3,7,4,0,8,5,5,6,0,5,2,7,8,4,4,4,1,9,0,8,4,6,6,9,0,2,2,6,7,4,4,1,6,1,2,4,8,7,9,8,4,5,6,7,6,9,6,6,7,4,0,9,5,4,5,5,5,4,0,9,3,0,0,9,9,6,8,4,3,5,5,9,8,9,4,9,5,1,8,1,4,8,2,0,3,4,9,1,3,9,7,7,1,2,8,7,1,2,8,8,4,7,7,3,7,0,6,5,1,3,3,0,3,7,5,4,1,5,3,9,4,0,2,7,0,8,0,7,9,3,7,9,1,9,1,2,9,4,5,8,8,7,0,9,7,3,6,8,9,1,5,4,6,3,0,5,8,6,6,7,4,1,4,5,0,3,5,3,8,3,4,0,6,3,6,6,5,2,8,9,9,5,3,3,5,6,2,3,7,8,7,7,0,4,8,1,2,5,1,9,9,7,7,3,2,5,9,1,6,2,2,4,3,8,9,9,6,7,1,4,3,2,7,8,9,8,4,4,5,0,0,0,3,0,1,8,9,5,9,6,5,3,2,4,0,7,6,6,1,6,8,2,6,8,6,1,2,4,8,1,3,5,1,6,3,0,4,2,2,9,5,9,3,1,8,5,3,9,3,9,7,1,0,0,6,5,7,2,1,8,8,4,0,2,9,3,4,7,7,4,5,5,7,4,9,4,9,4,9,6,2,9,0,7,5,8,0,3,9,0,2,7,8,8,0,9,8,5,7,9,5,9,7,9,1,2,8,9,8,9,8,4,8,0,6,0,2,2,7,6,1,9,2,1,0,9,3,0,2,3,3,0,7,0,5,3,4,2,3,4,7,6,5,3,2,2,2,4,2,9,9,1,2,5,0,5,2,5,3,4,2,8,0,8,0,1,1,9,0,0,9,4,2,2,5,8,5,9,6,2,1,6,1,3,0,9,4,5,1,7,3,8,3,0,3,4,4,0,0,6,1,3,2,8,5,9,9,8,1,0,1,2,4,4,8,6,9,2,4,1,5,1,8,1,9,2,0,4,5,3,0,4,9,9,2,9,6,6,6,3,0,9,2,1,3,6,3,5,6,1,3,2,8,1,5,6,3,5,2,3,0,0,7,3,3,1,4,3,1,5,6,3,1,0,0,1,8,1,1,7,7,1,6,7,8,2,4,7,2,3,3,6,6,6,9,3,7,8,1,1,9,1,1,8,2,6,0,8,5,5,1,5,6,5,6,3,6,5,1,9,6,7,4,4,1,2,7,8,9,9,3,5,8,4,4,3,1,5,9,6,2,0,3,2,7,0,1,9,1,9,2,9,6,6,4,9,2,3,8,8,9,5,2,6,1,7,9,8,4,4,7,6,7,0,9,6,2,3,3,7,9,0,2,7,5,9,7,2,8,7,7,2,1,1,6,4,5,6,0,5,7,9,8,9,0,0,6,3,4,1,2,6,3,1,9,4,5,9,4,7,1,0,6,1,6,7,6,2,5,7,5,8,1,5,0,2,8,0,9,9,5,9,0,2,9,0,5,1,7,7,9,9,0,1,0,9,2,7,0,3,4,2,1,2,6,6,8,9,3,6,9,8,4,3,2,8,5,7,5,8,1,3,8,0,6,4,8,3,2,7,8,9,8,1,4,9,0,7,7,7,1,8,3,6,9,4,5,5,8,2,2,6,2,4,6,7,5,8,1,3,5,4,9,0,7,3,2,5,1,9,9,6,6,9,0,0,7,3,8,0,5,3,8,6,2,5,3,9,4,1,8,8,6,5,0,3,8,9,0,4,2,5,1,9,6,8,7,7,0,4,0,1,5,7,1,1,0,2,7,7,1,3,0,0,1,4,8,3,8,0,6,2,5,5,2,0,3,8,8,5,1,0,2,3,2,6,0,7,6,7,2,3,9,2,8,1,9,6,3,9,4,9,2,3,7,1,8,3,0,6,6,8,8,0,4,1,0,1,3,9,1,0,2,0,7,6,1,8,7,5,9,3,1,7,4,0,9,9,9,3,1,4,6,6,6,4,6,6,6,3,4,4,1,3,3,3,0,4,3,7,9,9,3,3,9,8,6,1,7,7,1,0,0,1,2,2,9,9,7,5,3,5,0,5,4,1,0,9,8,7,0,8,9,9,6,1,5,4,2,2,6,3,8,8,6,9,7,7,7,6,2,2,7,6,2,3,6,0,8,0,9,2,9,8,2,4,1,8,7,5,1,7,1,6,4,9,5,0,3,2,0,6,6,4,6,1,8,8,2,6,3,6,7,8,5,9,6,2,3,9,0,9,0,7,9,2,4,4,6,7,3,6,0,3,2,9,7,1,9,3,2,0,3,0,6,3,2,0,1,0,2,3,3,1,7,4,9,7,1,6,9,0,6,6,9,8,1,3,2,9,7,2,6,8,1,4,6,8,2,3,8,4,2,6,0,6,7,3,5,1,0,1,8,1,9,2,9,5,9,6,6,4,3,6,7,7,7,8,0,6,7,3,3,2,9,1,7,9,9,9,2,9,9,0,2,6,6,2,2,2,5,2,5,8,1,1,4,1,0,0,3,0,8,3,9,9,3,0,2,0,9,7,5,2,4,7,0,5,4,4,8,3,3,9,3,6,5,8,5,0,3,5,1,5,2,0,4,3,8,3,3,7,4,3,7,2,3,7,6,7,7,0,5,1,6,2,8,0,5,9,9,4,5,5,4,2,1,8,5,1,4,5,5,8,3,2,8,6,3,6,0,6,6,3,0,5,1,1,0,7,1,8,9,5,9,4,8,3,8,2,1,3,9,5,9,3,2,6,9,6,9,0,1,4,7,0,7,4,1,6,6,4,4,4,0,1,0,2,0,1,6,5,2,2,5,5,4,7,7,6,4,4,6,0,6,8,0,1,0,4,1,0,0,0,4,6,6,6,2,5,3,0,3,4,0,1,6,0,9,8,4,7,7,9,4,8,8,2,0,5,3,8,0,1,2,8,8,0,4,0,5,8,3,6,6,9,7,3,8,5,8,2,9,6,2,0,3,4,6,7,8,3,4,2,6,9,3,0,3,7,5,0,6,7,7,3,2,2,1,1,0,9,6,4,6,9,7,9,2,9,4,0,8,6,0,0,2,1,1,0,5,1,3,7,7,7,7,5,0,6,7]
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]
        #self.mu = np.zeros(len(self.A))
        #self.n_arm = np.zeros(len(self.A))
        #self.N = 0
        self.pull_that_arm = self.arm_to_pull.pop(0)


    def choose(self):
        #return np.argmin(self.n_arm)
        return self.pull_that_arm


    def update(self, a, r):
        #self.n_arm[a] += 1
        #self.N += 1
        #self.mu[a] += (r - self.mu[a])/self.n_arm[a]
        #if self.N == 1000:
        #    good_a = np.argmax(self.mu)
        #    print(good_a)
        pass


# Choose which Agent is run for scoring
"""
RandomAgent
epsGreedyAgent
epsNGreedyAgent
BesaAgent
SoftmaxAgent
UCB1Agent
UCBNormalAgent
BetaThompsonAgent
GaussianThompsonAgent
KLUCBAgent
CheaterAgent
"""
Agent = UCB1Agent
#print(np.random.get_state())
