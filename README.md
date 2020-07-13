# Particle-Swarm-Optimization
My own implementation of the Particle Swarm Optimization (PSO) algorithm. 
PSO is an algorithm that generates a number of particles (Particle Swarm) in a function space that works together to find the global minimum. 
What makes PSO unique, is that it works on non-differentiable functions as it does not use differentiability of the function to find the desired solution. 
The PSO is a metaheuristic and does not guarantee optimality due to particles being stuck in local minimum values.
A more detailed description can be found on [here](https://en.wikipedia.org/wiki/Particle_swarm_optimization).

Let's see how the implementation works on the function ```0.02x^4 - 0.6x^3 - 35x^2 + 50x```. 
To generate the Particle Swarm, the first step is to define a particle class:
```python3
class Particle:
    def __init__(self, velocity, position, weights):
        self.velocity = velocity
        self.position = position
        self.loss = np.inf
        self.best_position = position
        self.weights = weights
    
    def move(self):
        new_position = self.position + self.velocity
        return new_position
    
    def Print_info(self):
        print('Velocity: {}'.format(self.velocity))
        print('Position: {}'.format(self.position))
    
    def update_velocity(self, best_particle):
        global_best = self.weights[0] * random.random() * (best_particle.position - self.position)
        known_best = self.weights[1] * random.random() * (self.best_position - self.position)
        new_velocity = self.weights[2] * self.velocity + global_best + known_best
        return new_velocity
```

Every particle is stored in the ParticleSwarm class. This class contains a bunch of helper functions to generate a GIF and optimize the swarm.

Let's see how some of these optimizations look with different weights:

![](PSO2.gif)

![](PSO3.gif)

![](PSO.gif)



















