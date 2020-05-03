import sys
import math
import queue
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

EPS = 1e-5

class Node:
    COUNT = 0

    def __init__(self, w, A, b, nElements, type='O', level=0):
        self.w = w
        self.A = A
        self.b = b
        self.nElements = nElements
        self.type = type  # L, R or O for left, right and root
        self.level = level
        self.upper_bound = 0
        self.lower_bound = -math.inf
        self.solution = None
        self.solve()
        Node.COUNT += 1

    def solve(self):
        print('=> solving')
        self.solution = optimize.linprog(self.w, A_ub=self.A, b_ub=self.b, bounds=(0, 1))
        print(self.solution)

        self.lower_bound = self.solution.fun

        if self.is_integer():
            self.upper_bound = self.solution.fun

    def is_integer(self):
        return np.all((self.solution.x < EPS) | (1 - EPS < self.solution.x))

    def identify_critic_rows(self, i):
        x = self.solution.x
        A = self.A

        # highly unoptimized for the sake of clarity
        for j in range(i + 1, len(x)):
            for e in range(0, self.nElements):
                for f in range(0, self.nElements):
                    if e == f:
                        continue

                    if A[e, i] != A[e, j] and A[f, i] == 1 and A[f, j] == 1 and x[j] > EPS:
                        return e, f, j

        print(A)

    def branch_novel(self):
        # get first fractional variable
        x = self.solution.x
        i = np.argmax((EPS < x) & (x < 1 - EPS))

        e, f, j = self.identify_critic_rows(i)

        nodes = []

        A = self.A

        new_row_A = np.zeros((1, A.shape[1]))
        for k in range(0, len(x)):
            if A[e, k] == 1 and A[f, k] == 1:
                new_row_A[-1, k] = 1

        # e and f have to be covered by a different subset on left branch
        new_A = np.zeros((A.shape[0] + 1, A.shape[1]))
        new_b = np.zeros(len(self.b) + 1)
        new_A[:-1, :] = A
        new_A[-1, :] = new_row_A
        new_b[:-1] = self.b
        # implicit new_b[-1] = 0

        node = Node(self.w, new_A, new_b, self.nElements, 'L', self.level + 1)
        nodes.append(node)

        # e and f have to be covered by the same subset on right branch
        new_A = np.zeros((A.shape[0] + 1, A.shape[1]))
        new_b = np.zeros(len(self.b) + 1)
        new_A[:-1, :] = A
        new_A[-1, :] = -new_row_A
        new_b[:-1] = self.b
        new_b[-1] = -1

        node = Node(self.w, new_A, new_b, self.nElements, 'R', self.level + 1)
        nodes.append(node)

        return nodes

    def branch_conventional(self):
        # get first fractional variable
        x = self.solution.x
        i = np.argmax((EPS < x) & (x < 1 - EPS))

        print('branching on i:%d  x_i: %f' % (i, x[i]))

        nodes = []

        # round down  (x_i = 0)
        new_A = np.zeros((self.A.shape[0] + 1, self.A.shape[1]))
        new_b = np.zeros(len(self.b) + 1)
        new_A[:-1, :] = self.A
        new_A[-1, i] = 1
        new_b[:-1] = self.b
        # implicit new_b[-1] = 0

        print('x_i = 0')
        node = Node(self.w, new_A, new_b, self.nElements, 'L', self.level + 1)
        nodes.append(node)

        # round up  (x_i = 1)
        new_A = np.zeros((self.A.shape[0] + 1, self.A.shape[1]))
        new_b = np.zeros(len(self.b) + 1)
        new_A[:-1, :] = self.A
        new_A[-1, i] = -1
        new_b[:-1] = self.b
        new_b[-1] = -1

        print('x_i = 1')
        node = Node(self.w, new_A, new_b, self.nElements, 'R', self.level + 1)
        nodes.append(node)

        return nodes

    def branch(self):
        return self.branch_novel()
        # return self.branch_conventional()

    def __lt__(self, other):
        return self.lower_bound < other.lower_bound


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=300, threshold=sys.maxsize)

    np.random.seed(42)
    max_experiment = 100

    node_counts = np.zeros((max_experiment, 2))

    experiment_no = 0
    while experiment_no < max_experiment:
        n_vars = np.random.randint(50, 100)
        n_elements = np.random.randint(n_vars, n_vars+100)
        w = -np.random.uniform(0.5, 1, n_vars)
        A = (np.random.uniform(0, 1, [n_elements, n_vars]) < 0.5).astype(float)
        b = np.ones(n_elements)

        solutions = np.zeros((2, n_vars))

        for method in [0, 1]:
            root = Node(w, A, b, n_elements)
            if root.is_integer():
                experiment_no -= 1
                break

            q = queue.PriorityQueue()
            q.put(root)

            best_bound = 0
            best_node = None

            while not q.empty():
                node = q.get()
                print('====\n going into: %d - %s\n====' % (node.level, node.type))
                if node.is_integer() or node.lower_bound > best_bound or node.solution.success == False:
                    continue

                if method == 0:
                    nodes = node.branch_conventional()
                else:
                    nodes = node.branch_novel()

                for n in nodes:
                    if n.is_integer():
                        print('====\n Found an integer solution\n====')
                        if n.solution.fun < best_bound:
                            best_bound = n.solution.fun
                            best_node = n
                            print('====\n A better solution: %f\n====' % best_bound)

                    if n.lower_bound < best_bound:
                        q.put(n)

            print('best')
            print(best_node.solution)

            print('total node count: ', Node.COUNT)
            node_counts[experiment_no, method] = Node.COUNT
            solutions[method, :] = best_node.solution.x
            Node.COUNT = 0

        experiment_no += 1

    if np.sum(np.abs(solutions[0, :] - solutions[1, :])) > 1e-5:
        raise Exception('two methods do not agree')

    print('node counts\n', node_counts)


if __name__ == "__main__":
    main()
