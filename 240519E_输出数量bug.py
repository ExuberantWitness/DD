from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
import numpy as np

class MultiObjectiveMixedVariableProblem(ElementwiseProblem):
    def __init__(self, **kwargs):
        vars = {
            "b": Binary(),
            "x": Choice(options=["nothing", "multiply"]),
            "y": Integer(bounds=(-2, 2)),
            "z": Real(bounds=(-5, 5)),
        }
        super().__init__(vars=vars, n_obj=2, n_ieq_constr=0, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        b, x, y, z = X["b"], X["x"], X["y"], X["z"]

        f1 = z ** 2 + y ** 2
        f2 = (z + 2) ** 2 + (y - 1) ** 2

        if b:
            f2 = 100 * f2

        if x == "multiply":
            f2 = 10 * f2

        out["F"] = [f1, f2]


# Custom callback to capture all evaluated solutions
class CaptureCallback:
    def __init__(self):
        self.data = []

    def __call__(self, algorithm):
        # Append evaluated solutions and their objectives to the data list
        for ind in algorithm.pop:
            self.data.append((ind.get("X"), ind.get("F")))


# Instantiate the callback
capture_callback = CaptureCallback()

# Create the optimization problem
problem = MultiObjectiveMixedVariableProblem()

# Define the optimization algorithm
algorithm = MixedVariableGA(pop_size=300, survival=RankAndCrowdingSurvival())

# Run the optimization with the callback
res = minimize(problem,
               algorithm,
               ('n_gen', 50),
               seed=1,
               verbose=False,
               callback=capture_callback)

# Extract all evaluated points from the callback
evaluated_X = [x for x, _ in capture_callback.data]
evaluated_F = [f for _, f in capture_callback.data]







# Convert lists to numpy arrays
evaluated_X = np.array(evaluated_X)
evaluated_F = np.array(evaluated_F)
for i in range(evaluated_F.shape[0]):
    TEMP_V= evaluated_F[i,:]
    if np.max(TEMP_V)>10:
        evaluated_F[i, 0] =0
        evaluated_F[i, 1] = 0


# Print all evaluated solutions and their corresponding objective values

print("All Evaluated Objective Values:")
print(evaluated_F)

# Plot all evaluated points
plot = Scatter()
plot.add(evaluated_F, facecolor="none", edgecolor="blue", label="All Evaluated Points")
plot.show()
