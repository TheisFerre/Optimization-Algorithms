import numpy as np
import matplotlib.pyplot as plt


class Particle:
    def __init__(self, position):
        self.position = position
        self.loss = np.inf
        self.best_position = position


class DifferentialEvolution:
    """
    Differential Evolution class
    """

    def __init__(self, obj_func, bounds, iterations,
                 particles=10, mutation=0.5, recombination=0.5):

        self.obj_func = obj_func
        self.iterations = iterations
        self.mutation = mutation
        self.recombination = recombination
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
            position = np.array([np.random.uniform(dim[0], dim[1])
                                 for dim in self.bounds])
            particles.append(Particle(position))
        return particles

    def objective_function(self, position):
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
            particle.loss = self.objective_function(particle.position)
            if particle.loss < best_loss:
                best_part = particle
                best_loss = best_part.loss
        return best_part

    def isvalid(self, position):
        """
        Check if particle is within bounds
        """
        for idx, value in enumerate(position):
            if value < self.bounds[idx][0] or value > self.bounds[idx][1]:
                return False
        return True

    def mutate(self, particle, choices):

        # Mutate
        donor_vector = self.particles[choices[0]].position + self.mutation * \
        (self.particles[choices[1]].position - self.particles[choices[2]].position)

        i_rand = np.random.randint(0, len(donor_vector))

        # Recombination
        proposal_position = np.ones(donor_vector.shape)
        for j, value in enumerate(donor_vector):
            if np.random.random() <= self.recombination or j == i_rand:
                proposal_position[j] = value
            else:
                proposal_position[j] = particle.position[j]

        # Selection
        if self.objective_function(proposal_position) <= self.objective_function(particle.position) and self.isvalid(proposal_position):
            next_gen_pos = proposal_position
        else:
            next_gen_pos = particle.position
        
        return next_gen_pos

    def optimize(self, record=False):
        """
        Optimize swarm
        """
        best_part = self.get_best_particle()
        best_loss = best_part.loss
        if record:
            loss_list = [0] * self.iterations

        for i in range(self.iterations):
            next_gen_positions = [0] * len(self.particles)
            for idx, particle in enumerate(self.particles):

                indices = np.array(list(range(len(self.particles))))
                choices = np.random.choice(
                    np.delete(indices, idx), size=3, replace=False)

                next_gen_positions[idx] = self.mutate(particle, choices)

                particle.loss = self.objective_function(particle.position)

                if particle.loss < self.objective_function(particle.best_position):
                    particle.best_position = particle.position

                if self.objective_function(particle.best_position) < best_loss:
                    best_part = particle
                    best_loss = particle.loss

            if record:
                loss_list[i] = best_loss
            
            #Update positions to next generation
            for idx, particle in enumerate(self.particles):
                particle.position = next_gen_positions[idx]

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
num_particles = 20
mutation = 0.9
recombination = 0.9

DE = DifferentialEvolution(obj_func1, bounds, num_iterations, num_particles, mutation=mutation, recombination=recombination)

DE.optimize()

#### TEST CASE 2 ####


def obj_func2(*args):
    """
    Rosenbrock function
    """

    return (1-args[0])**2 + 100*(args[1] - args[0]**2)**2


bounds = [(-50, 50), (-50, 50)]
num_iterations = 500
num_particles = 20
mutation = 0.9
recombination = 0.9

DE = DifferentialEvolution(obj_func2, bounds, num_iterations,
                      num_particles, mutation, recombination)

loss_list = DE.optimize(record=True)

plt.plot(loss_list)
plt.show()


#### TEST CASE 3 ####


def obj_func3(*args):
    """
    Ackley's function
    """

    return 20 - 20*np.exp(-0.2 * np.sqrt(1/2 * (args[0]**2 + args[1]**2))) - np.exp(1/2 * (np.cos(2*np.pi*args[0]) + np.cos(2*np.pi*args[1])))


bounds = [(-50, 50), (-50, 50)]
num_iterations = 500
num_particles = 20
mutation = 0.9
recombination = 0.9

DE = DifferentialEvolution(obj_func3, bounds, num_iterations,
                      num_particles, mutation, recombination)

loss_list = DE.optimize(record=True)

plt.plot(loss_list)
plt.show()
