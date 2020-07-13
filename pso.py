import numpy as np
import matplotlib.pyplot as plt

import numpy as np


class Particle:
    def __init__(self, velocity, position, weights):
        self.velocity = velocity
        self.position = position
        self.loss = np.inf
        self.best_position = position
        self.weights = weights

    def move(self):
        """
        Move particle to a new position
        """
        new_position = self.position + self.velocity
        return new_position

    def Print_info(self):
        """
        print particle information
        """
        print('Velocity: {}'.format(self.velocity))
        print('Position: {}'.format(self.position))

    def update_velocity(self, best_particle):
        """
        Update velocity of particle
        """
        global_best = self.weights[0] * np.random.random() * \
            (best_particle.position - self.position)

        known_best = self.weights[1] * np.random.random() * \
            (self.best_position - self.position)

        new_velocity = self.weights[2] * \
            self.velocity + global_best + known_best

        return new_velocity


class ParticleSwarm:
    """
    Particle swarm Class
    """

    def __init__(self, obj_func, bounds, iterations,
                 particles=10, particle_weights=(2, 1, 0.5)):

        self.obj_func = obj_func
        self.iterations = iterations
        self.weights = particle_weights
        self.num_particles = particles
        self._particles = None
        self.bounds = bounds

    @property
    def particles(self):

        if self._particles is None:
            self._particles = self.create_particles(self.num_particles)
        return self._particles

    def create_particles(self, num_particles):
        """
        Helper function for creating particles
        """
        particles = []
        for _ in range(self.num_particles):
            velocity = np.random.uniform(size=len(self.bounds))
            position = np.array([np.random.uniform(dim[0], dim[1])
                                 for dim in self.bounds])
            weights = self.weights
            particles.append(Particle(velocity, position, weights))
        return particles

    def loss_particle(self, position):
        """
        Compute function value for a position
        """
        objective_value = self.obj_func(*position)
        return objective_value

    def get_best_particle(self):
        """
        Finding the best positioned particle
        """
        best_loss = np.inf
        best_part = None
        for particle in self.particles:
            particle.loss = self.loss_particle(particle.position)
            if particle.loss < best_loss:
                best_part = particle
                best_loss = best_part.loss
        return best_part

    def isvalid(self, particle):
        """
        Check if particle is within bounds
        """
        pos = particle.position
        for idx, value in enumerate(pos):
            if value < self.bounds[idx][0] or value > self.bounds[idx][1]:
                return False
        return True

    def optimize(self, record=False):
        """
        Optimize swarm
        """
        best_part = self.get_best_particle()
        best_loss = best_part.loss
        if record:
            loss_list = [0] * self.iterations

        for i in range(self.iterations):
            for particle in self.particles:

                particle.velocity = particle.update_velocity(best_part)
                particle.position = particle.move()

                if self.isvalid(particle):
                    particle.loss = self.loss_particle(particle.position)

                    if particle.loss < self.loss_particle(particle.best_position):
                        particle.best_position = particle.position

                    if self.loss_particle(particle.best_position) < best_loss:
                        best_part = particle
                        best_loss = particle.loss

            if record:
                loss_list[i] = best_loss

            print(f'Iteration {i+1}')
            print(f'Best particle found at position: {best_part.position}')
            print(f'Lowest objective value: {best_loss}')
            print('#'*8)

        if record:
            return loss_list


#### TEST CASE 1 ####


def obj_func1(*args):

    return 0.2*args[0]**3 * args[0]/np.sin(args[0])


bounds = [(-5, 5)]
num_iterations = 50
num_particles = 2000
weights = (2, 1, 0.5)

swarm = ParticleSwarm(obj_func1, bounds, num_iterations,
                      num_particles, weights)

swarm.optimize()

#### TEST CASE 2 ####


def obj_func2(*args):
    """
    Rosenbrock function
    """

    return (1-args[0])**2 + 100*(args[1] - args[0]**2)**2


bounds = [(-50, 50), (-50, 50)]
num_iterations = 50
num_particles = 2000
weights = (2, 1, 0.5)

swarm = ParticleSwarm(obj_func2, bounds, num_iterations,
                      num_particles, weights)

loss_list = swarm.optimize(record=True)

plt.plot(loss_list)
plt.show()
