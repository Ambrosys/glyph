import glyph.application as application
from glyph.assessment import AAssessmentRunner


class AssessmentRunnerMock(AAssessmentRunner):
    """A mock assessment runner. Gives fitness values (1.0, 1.0) to every indidvidual."""

    def measure(self, individual):
        return 1.0, 1.0


def test_gp_runner_reproducibility(SympyIndividual):

    def run(pop_size, num_generations):
        gp_runner = application.default_gprunner(SympyIndividual, AssessmentRunnerMock())
        gp_runner.init(pop_size)
        for _ in range(num_generations):
            gp_runner.step()
        return gp_runner

    import random
    seed = 1234567890
    pop_size = 10
    num_generations = 4

    random.seed(seed)
    gp_runner_1 = run(pop_size, num_generations)
    random.seed(seed)
    gp_runner_2 = run(pop_size, num_generations)
    assert gp_runner_1.population == gp_runner_2.population
    fit_vals_1 = [ind.fitness.values for ind in gp_runner_1.population]
    fit_vals_2 = [ind.fitness.values for ind in gp_runner_2.population]
    assert fit_vals_1 == fit_vals_2
    assert gp_runner_1.halloffame[:] == gp_runner_2.halloffame[:]
    assert gp_runner_1.logbook == gp_runner_2.logbook
