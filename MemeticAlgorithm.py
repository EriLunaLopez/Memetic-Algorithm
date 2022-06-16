from functools import partial
from itertools import count
from typing import List, Tuple
from random import choices, randint, randrange
from typing import List, Callable

import math
import random


# Provides the methods to create and solve the firefighter problem
class FFP:

  # Constructor
  #   fileName = The name of the file that contains the FFP instance
  def __init__(self, fileName):
    file = open(fileName, "r")    
    text = file.read()    
    tokens = text.split()
    seed = int(tokens.pop(0))
    self.n = int(tokens.pop(0))
    model = int(tokens.pop(0))  
    int(tokens.pop(0)) # Ignored
    # self.state contains the state of each node
    #    -1 On fire
    #     0 Available for analysis
    #     1 Protected
    self.state = [0] * self.n
    nbBurning = int(tokens.pop(0))
    for i in range(nbBurning):
      b = int(tokens.pop(0))
      self.state[b] = -1      
    self.graph = []    
    for i in range(self.n):
      self.graph.append([0] * self.n);
    while tokens:
      x = int(tokens.pop(0))
      y = int(tokens.pop(0))
      self.graph[x][y] = 1
      self.graph[y][x] = 1    

  # Solves the FFP by using a given method and a number of firefighters
  #   method = Either a string with the name of one available heuristic or an object of class HyperHeuristic
  #   nbFighters = The number of available firefighters per turn
  #   debug = A flag to indicate if debugging messages are shown or not
  def solve(self, Genome, nbFighters, debug = False):
    spreading = True
    if (debug):
      print("Initial state:" + str(self.state))    
    t = 0
    contador=0
    while (spreading):
      if (debug):
        print("Features")
        print("")
        print("Graph density: %1.4f" % (self.getFeature("EDGE_DENSITY")))
        print("Average degree: %1.4f" % (self.getFeature("AVG_DEGREE")))
        print("Burning nodes: %1.4f" % self.getFeature("BURNING_NODES"))
        print("Burning edges: %1.4f" % self.getFeature("BURNING_EDGES"))
        print("Nodes in danger: %1.4f" % self.getFeature("NODES_IN_DANGER"))
      # It protects the nodes (based on the number of available firefighters)
      for i in range(nbFighters):
        if Genome[contador]==0:
            heuristic = "GDEG"
            contador += 1
        else: 
            heuristic = "LDEG" 
            contador +=1
        node = self.__nextNode(heuristic)
        if (node >= 0):
          # The node is protected   
          self.state[node] = 1
          # The node is disconnected from the rest of the graph
          for j in range(len(self.graph[node])):
            self.graph[node][j] = 0
            self.graph[j][node] = 0
          if (debug):
            print("\tt" + str(t) + ": A firefighter protects node " + str(node))            
      # It spreads the fire among the unprotected nodes
      spreading = False 
      state = self.state.copy()
      for i in range(len(state)):
        # If the node is on fire, the fire propagates among its neighbors
        if (state[i] == -1): 
          for j in range(len(self.graph[i])):
            if (self.graph[i][j] == 1 and state[j] == 0):
              spreading = True
              # The neighbor is also on fire
              self.state[j] = -1
              # The edge between the nodes is removed (it will no longer be used)
              self.graph[i][j] = 0
              self.graph[j][i] = 0
              if (debug):
                print("\tt" + str(t) + ": Fire spreads to node " + str(j))     
      t = t + 1
      if (debug):
        print("---------------")
    if (debug):    
      print("Final state: " + str(self.state))
      print("Solution evaluation: " + str(self.getFeature("BURNING_NODES")))
    return self.getFeature("BURNING_NODES")

  # Selects the next node to protect by a firefighter
  #   heuristic = A string with the name of one available heuristic
  def __nextNode(self, heuristic):
    index  = -1
    best = -1
    for i in range(len(self.state)):
      if (self.state[i] == 0):
        index = i        
        break
    value = -1
    for i in range(len(self.state)):
      if (self.state[i] == 0):
        if (heuristic == "LDEG"):
          # It prefers the node with the largest degree, but it only considers
          # the nodes directly connected to a node on fire
          for j in range(len(self.graph[i])):
            if (self.graph[i][j] == 1 and self.state[j] == -1):
              value = sum(self.graph[i])              
              break
        elif (heuristic == "GDEG"):        
          value = sum(self.graph[i])          
        else:
          print("=====================")
          print("Critical error at FFP.__nextNode.")
          print("Heuristic " + heuristic + " is not recognized by the system.")          
          print("The system will halt.")
          print("=====================")
          exit(0)
      if (value > best):
        best = value
        index = i
    return index

  # Returns the value of the feature provided as argument
  #   feature = A string with the name of one available feature
  def getFeature(self, feature):
    f = 0
    if (feature == "EDGE_DENSITY"):
      n = len(self.graph)      
      for i in range(len(self.graph)):
        f = f + sum(self.graph[i])
      f = f / (n * (n - 1))
    elif (feature == "AVG_DEGREE"):
      n = len(self.graph) 
      count = 0
      for i in range(len(self.state)):
        if (self.state[i] == 0):
          f += sum(self.graph[i])
          count += 1
      if (count > 0):
        f /= count
        f /= (n - 1)
      else:
        f = 0
    elif (feature == "BURNING_NODES"):
      for i in range(len(self.state)):
        if (self.state[i] == -1):
          f += 1
      f = f / len(self.state)
    elif (feature == "BURNING_EDGES"):
      n = len(self.graph) 
      for i in range(len(self.graph)):
        for j in range(len(self.graph[i])):
          if (self.state[i] == -1 and self.graph[i][j] == 1):
            f += 1
      f = f / (n * (n - 1))    
    elif  (feature == "NODES_IN_DANGER"):
      for j in range(len(self.state)):
        for i in range(len(self.state)):
          if (self.state[i] == -1 and self.graph[i][j] == 1):
            f += 1
            break
      f /= len(self.state)
    else:      
      print("=====================")
      print("Critical error at FFP._getFeature.")
      print("Feature " + feature + " is not recognized by the system.")          
      print("The system will halt.")
      print("=====================")
      exit(0)
    return f

  # Returns the string representation of this problem
  def __str__(self):
    text = "n = " + str(self.n) + "\n"
    text += "state = " + str(self.state) + "\n"
    for i in range(self.n):
      for j in range(self.n):
        if (self.graph[i][j] == 1 and i < j):
          text += "\t" + str(i) + " - " + str(j) + "\n"
    return text

# Provides the methods to create and use hyper-heuristics for the FFP
# This is a class you must extend it to provide the actual implementation
class HyperHeuristic:

  # Constructor
  #   features = A list with the names of the features to be used by this hyper-heuristic
  #   heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
  def __init__(self, features, heuristics):
    if (features):
      self.features = features.copy()
   # else:
    #  print("=====================")
     # print("Critical error at HyperHeuristic.__init__.")
      #print("The list of features cannot be empty.")
      #print("The system will halt.")
      #print("=====================")
      #exit(0)
    if (heuristics):
      self.heuristics = heuristics.copy()
    else:
      print("=====================")
      print("Critical error at HyperHeuristic.__init__.")
      print("The list of heuristics cannot be empty.")
      print("The system will halt.")
      print("=====================")
      exit(0)
  
  # Returns the next heuristic to use
  #   problem = The FFP instance being solved
  def nextHeuristic(self, problem):
    print("=====================")
    print("Critical error at HyperHeuristic.nextHeuristic.")
    print("The method has not been overriden by a valid subclass.")
    print("The system will halt.")
    print("=====================")
    exit(0)

  # Returns the string representation of this hyper-heuristic 
  def __str__(self):
    print("=====================")
    print("Critical error at HyperHeuristic.__str__.")
    print("The method has not been overriden by a valid subclass.")
    print("The system will halt.")
    print("=====================")
    exit(0)

# A dummy hyper-heuristic for testing purposes.
# The hyper-heuristic creates a set of randomly initialized rules.
# Then, when called, it measures the distance between the current state and the
# conditions in the rules
# The rule with the condition closest to the problem state is the one that fires
class DummyHyperHeuristic(HyperHeuristic):

  # Constructor
  #   features = A list with the names of the features to be used by this hyper-heuristic
  #   heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
  #   nbRules = The number of rules to be contained in this hyper-heuristic  
  def __init__(self, features, heuristics, nbRules, seed):
    super().__init__(features, heuristics)
    random.seed(seed)
    self.conditions = []
    self.actions = []
    for i in range(nbRules):
      self.conditions.append([0] * len(features))
      for j in range(len(features)):
        self.conditions[i][j] = random.random()
      self.actions.append(heuristics[random.randint(0, len(heuristics) - 1)])
  
  # Returns the next heuristic to use
  #   problem = The FFP instance being solved
  def nextHeuristic(self, problem):
    minDistance = float("inf")
    index = -1
    state = []
    for i in range(len(self.features)):
      state.append(problem.getFeature(self.features[i]))
    print("\t" + str(state))
    for i in range(len(self.conditions)):
      distance = self.__distance(self.conditions[i], state)      
      if (distance < minDistance):
        minDistance = distance
        index = i
    heuristic = self.actions[index] 
    print("\t\t=> " + str(heuristic) + " (R" + str(index) + ")")
    return heuristic

  # Returns the string representation of this dummy hyper-heuristic
  def __str__(self):
    text = "Features:\n\t" + str(self.features) + "\nHeuristics:\n\t" + str(self.heuristics) + "\nRules:\n"
    for i in range(len(self.conditions)):      
      text += "\t" + str(self.conditions[i]) + " => " + self.actions[i] + "\n"      
    return text

  # Returns the Euclidian distance between two vectors
  def __distance(self, vectorA, vectorB):
    distance = 0
    for i in range(len(vectorA)):
      distance += (vectorA[i] - vectorB[i]) ** 2
    distance = math.sqrt(distance)
    return distance

class SequHyperHeuristic(HyperHeuristic):

# 100101 = LDEG, GDEG, GDEG, LDEG, GDEG, LDEG
# 100101 = LDEG, GDEG, GDEG, LDEG, GDEG, LDEG
# 100101 = LDEG, GDEG, GDEG, LDEG, GDEG, LDEG
# 100101 = LDEG, GDEG, GDEG, LDEG, GDEG, LDEG

  def __init__(self, heuristics, nbBits, seed):
    super().__init__([],heuristics)
    random.seed(seed)
    self.sequence = [0] * nbBits
    for i in range(nbBits):
      if (random.random()> 0.5):
        self.sequence[i]= 1
    self.index = 0

  def nextHeuristic(self, problem):
    heuristic = "LDEG"
    if (self.sequence[self.index]== 0):
      heuristic = "GDEG"
    self.index += 1
    if (self.index >= len(self.sequence)):
      self.index = 0
    print("\t\t>= " + str(heuristic))
    return heuristic

  def __str__(self):
    return str(self.sequence)
# Tests
# =====================

#fileName = "instances/BBGRL/100_ep0.1_0_gilbert_1.in"
#Solves the problem using heuristic LDEG and one firefighter
#problem = FFP(fileName)
#print("LDEG = " + str(problem.solve("LDEG", 1, False)))

# Solves the problem using heuristic GDEG and one firefighter
#problem = FFP(fileName)
#print("GDEG = " + str(problem.solve("GDEG", 1, False)))

#Solves the problem using a randomly generated dummy hyper-heuristic
#problem = FFP(fileName)
#seed = random.randint(0, 1000)
#print(seed)
#hh = DummyHyperHeuristic(["EDGE_DENSITY", "BURNING_NODES", "NODES_IN_DANGER"], ["LDEG", "GDEG"], 2, seed)
#print(hh)
#print("Dummy HH = " + str(problem.solve(hh, 1, False)))

#problem = FFP(fileName)
#seed = random.randint(0, 1000)
#print(seed)
#hh = SequHyperHeuristic( ["LDEG", "GDEG"], 4, seed)
#print(hh)
#print(" Test HH = " + str(problem.solve(hh, 1, False)))




Genome=List[int]
#print(Genome)
Population=List[Genome]
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
#Genetic representation of a solution
def generate_genome(length: int) -> Genome:
    return choices([0,1], k=length)


#A function to generate new solutions
def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]


#Fitness function to evaluate solutions
def fitness(genome: Genome) -> int:
    #fileName = "instances/GBRL/50_ep0.1_0_gilbert_1.in"
    fileName = "instances/BBGRL/1000_ep0.01_0_gilbert_1.in"
    problem = FFP(fileName)
    burning_nodes = problem.solve(genome, 1, False)
    #Regresar el valor de burning nodes 
    return burning_nodes


#Selection function
def selection_pair(population:Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(genome) for genome in population],
        k=2

    )

#Crossover function
def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of same length")
    
    length = len(a)
    if length < 2:
        return a, b
    
    p = randint(1, length -1)
    return a[0:p] + b[p:], b[0:p] + a[p:]

#Mutation
def mutation(genome: Genome, num: int =1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index]= genome[index] if random.randint(0,1) > probability else abs(genome[index]-1)
    return genome

def run_evolution(
         populate_func: PopulateFunc,
         fitness_func: FitnessFunc,
         fitness_limit: int,
         selection_func: SelectionFunc = selection_pair,
         crossover_func: CrossoverFunc = single_point_crossover,
         mutation_func: MutationFunc = mutation,
         generation_limit: int = 100,


) -> Tuple[Population, int]:
    population = populate_func()

    for i in range(generation_limit):
        population = sorted(
            population,
            key=lambda genome: fitness_func(genome),
            reverse=True
        )

        if fitness_func(population[0]) <= fitness_limit:
            break

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
        
        population = next_generation
    
    population = sorted(
        population,
        key=lambda genome: fitness_func(genome),
        reverse=True
    )

    return population, i

population, generations = run_evolution(
    populate_func=partial(
        generate_population, size=10, genome_length=10 
    ),
    fitness_func=partial(
        fitness, 
    ),
    fitness_limit=0.20, 
    generation_limit=100
)

print(f"number of generations: {generations}")
print(f"The best solution is: {(population[0])}")
fileName = "instances/GBRL/1000_ep0.01_0_gilbert_1.in"
#fileName = "instances/BBGRL/1000_ep0.01_0_gilbert_1.in"
problem = FFP(fileName)
print(f"The percentage of burning nodes is {(problem.solve(population[0],1,False))}")
