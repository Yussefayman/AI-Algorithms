from math import sqrt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import random
r = random.Random()
r.seed("AI")


import math


# region SearchAlgorithms
class Stack:

    def __init__(self):
        self.stack = []

    def push(self, value):
        if value not in self.stack:
            self.stack.append(value)
            return True
        else:
            return False

    def exists(self, value):
        if value not in self.stack:
            return True
        else:
            return False

    def pop(self):
        if len(self.stack) <= 0:
            return ("The Stack == empty")
        else:
            return self.stack.pop()

    def top(self):
        return self.stack[0]


class Node:
    id = None
    up = None
    down = None
    left = None
    right = None
    previousNode = None
    edgeCost = None
    gOfN = None  # total edge cost
    hOfN = None  # heuristic value
    heuristicFn = None
    indexRow = None
    indexCol = None

    def __init__(self, value):
        self.value = value
        self.previousNode=None
        self.edgeCost=0




class SearchAlgorithms:

    ''' * DON'T change Class, Function or Parameters Names and Order
        * You can add ANY extra functions,
          classes you need as long as the main
          structure is left as is '''
    path = []  # Represents the correct path from start node to the goal node.
    fullPath = []  # Represents all visited nodes from the start node to the goal node.
    totalCost = None
    Maze = None
    ''' mazeStr contains the full board
             The board is read row wise,
            the nodes are numbered 0-based starting
            the leftmost node'''
    def __init__(self, mazeStr,edgeCost = None):
        self.Maze = self.createBoard(mazeStr,edgeCost)

    def createBoard(self,maze_starter,edgeCost):
        board = []
        row = []
        rowsCount = 0
        colsCount = 0
        for i in maze_starter:
            if i == ' ':
                board.append(row)
                row = []
                rowsCount+=1
            elif i == ',':
                colsCount+=1
                continue
            else:
                row.append(i)

        rowsCount+=1
        board.append(row)
        colsCount = int((colsCount/rowsCount)+1)

        edgeCostVal = []
        MM = 0
        for i in range(0,rowsCount):
           container = []
           for j in range(MM,MM+colsCount):
               container.append(edgeCost[j])
           MM+=colsCount
           edgeCostVal.append(container)

        Maze = []
        for i in range(0,rowsCount):
            rowContainer = []
            for j in range(0,colsCount):
                curr_node = Node(board[i][j])
                curr_node.id = [i,j]
                curr_node.up =[i-1,j]
                curr_node.down=[i+1,j]
                curr_node.left=[i,j-1]
                curr_node.right=[i,j+1]
                curr_node.edgeCost=edgeCostVal[i][j]
                curr_node.heuristicFn=math.inf
                curr_node.gOfNFn = math.inf
                curr_node.hOfN = math.inf
                rowContainer.append(curr_node)
            Maze.append(rowContainer)
        #print(edgeCostVal)
        #print(board)
        return Maze

    def calcManhattan(self,p1,p2):
        return abs(p1[0]-p2[0])+abs(p1[1]-p2[1])

    def AstarManhattanHeuristic(self):
        open_list = []
        closed_list = []
        curr_node = None
        goal_node = None
        indexStart = None
        indexGoal = None
        rowsNum = len(self.Maze)
        colsNum = len(self.Maze[0])
        for i in range(0,len(self.Maze)):
            for j in range(0,len(self.Maze[0])):
                if self.Maze[i][j].value =='S':
                    curr_node=self.Maze[i][j]
                    indexStart=curr_node.id
                if self.Maze[i][j].value =='E':
                    goal_node =self.Maze[i][j]
                    indexGoal=goal_node.id

        #print(hofnVal)
        #print("index start",indexStart)
        #print("index goal ",indexGoal)
        gofnVal = curr_node.gOfN = 0
        hofnVal = curr_node.hOfN = self.calcManhattan(indexStart, indexGoal)
        curr_node.heuristicFn= gofnVal+hofnVal

        open_list.append(curr_node)
        while open_list:
            curr_node = min(open_list,key=lambda x: x.heuristicFn)
            self.fullPath.append((curr_node.id[0]*colsNum)+curr_node.id[1])
            if curr_node == goal_node:
                curr_node = goal_node
                while curr_node!=None:
                    xcoord = curr_node.id[0]
                    ycoord = curr_node.id[1]
                    self.path.append((colsNum*xcoord)+ycoord)
                    curr_node = self.Maze[curr_node.previousNode[0]][curr_node.previousNode[1]] if curr_node.previousNode!=None  else None
                self.path.reverse()
                break
            open_list.remove(curr_node)
            closed_list.append(curr_node)
            neigbours = [curr_node.up,curr_node.down,curr_node.left,curr_node.right]
            for neigbour in neigbours:
                if neigbour[0] not in range(0,len(self.Maze)) or neigbour[1] not in range(0,len(self.Maze[0])):
                    continue
                else:
                    #print(neigbour)
                    neigbourNode = self.Maze[neigbour[0]][neigbour[1]]
                    if neigbourNode.value == '#':
                        continue
                    if neigbourNode in closed_list:
                        continue
                    if neigbourNode in open_list:
                        for k in range(len(open_list)):
                            if open_list[k].id==neigbour:
                                newGofN = curr_node.gOfN+neigbourNode.edgeCost
                                newHofn = self.calcManhattan(indexGoal,neigbour)
                                newHeuristicofN = newGofN+newHofn
                                if open_list[k].heuristicFn > newHeuristicofN:
                                    open_list[k].gOfN = newGofN
                                    open_list[k].heuristicFn=newHeuristicofN
                                    open_list[k].previousNode = curr_node.id
                                    open_list[k].hOfN = newHofn
                                    self.Maze[neigbour[0]][neigbour[1]].gOfN = newGofN
                                    self.Maze[neigbour[0]][neigbour[1]].heuristicFn = newHeuristicofN
                                    self.Maze[neigbour[0]][neigbour[1]].previousNode = curr_node.id
                    else:
                        self.Maze[neigbour[0]][neigbour[1]].gOfN = curr_node.gOfN+neigbourNode.edgeCost
                        self.Maze[neigbour[0]][neigbour[1]].hOfN =self.calcManhattan(neigbour,indexGoal)
                        self.Maze[neigbour[0]][neigbour[1]].heuristicFn =self.Maze[neigbour[0]][neigbour[1]].gOfN +self.Maze[neigbour[0]][neigbour[1]].hOfN
                        self.Maze[neigbour[0]][neigbour[1]].previousNode = curr_node.id
                        open_list.append(neigbourNode)
        self.totalCost = self.Maze[indexGoal[0]][indexGoal[1]].gOfN
        return self.fullPath, self.path, self.totalCost

# endregion

# region KNN
class KNN_Algorithm:

    def __init__(self, K):
        self.K = K
        self.points_distances = []
        self.y_predict = []

    def euclidean_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1)-np.array(p2))


    def sort_list(self,listOFlist):
        listOFlist.sort(key = lambda x : x[1])
        return listOFlist

    def KNN(self, X_train, X_test, Y_train, Y_test):
        for i in range(0,len(X_test)):
            i_distances = []
            for j in range(0,len(X_train)):
                i_distances.append([Y_train[j],self.euclidean_distance(X_train[j],X_test[i])])
            self.sort_list(i_distances)
            self.points_distances.append(i_distances[0:self.K])
        #print(self.points_distances[:5])
        for i in self.points_distances:
            malignant=0
            benign=0
            for j in i:
                if j[0]==0:
                    benign+=1
                elif j[0]==1:
                    malignant+=1
            if malignant>benign:
                self.y_predict.append(1)
            else:
                self.y_predict.append(0)
        countSuccesPrediction = 0
        for i in range(0,len(Y_test)):
            if(Y_test[i]==self.y_predict[i]):
                countSuccesPrediction+=1
        accuracy = (countSuccesPrediction/len(Y_test))*100
        return accuracy


# endregion


# region GeneticAlgorithm
class GeneticAlgorithm:
    Cities = [1, 2, 3, 4, 5, 6]
    DNA_SIZE = len(Cities)
    POP_SIZE = 20
    GENERATIONS = 5000

    """
    - Chooses a random element from items, where items is a list of tuples in
       the form (item, weight).
    - weight determines the probability of choosing its respective item. 
     """

    def weighted_choice(self, items):
        weight_total = sum((item[1] for item in items))
        n = r.uniform(0, weight_total)
        for item, weight in items:
            if n < weight:
                return item
            n = n - weight
        return item

    """ 
      Return a random character between ASCII 32 and 126 (i.e. spaces, symbols, 
       letters, and digits). All characters returned will be nicely printable. 
    """

    def random_char(self):
        return chr(int(r.randrange(32, 126, 1)))

    """ 
       Return a list of POP_SIZE individuals, each randomly generated via iterating 
       DNA_SIZE times to generate a string of random characters with random_char(). 
    """

    def random_population(self):
        pop = []
        for i in range(1, 21):
            x = r.sample(self.Cities, len(self.Cities))
            if x not in pop:
                pop.append(x)
        return pop

    """ 
      For each gene in the DNA, this function calculates the difference between 
      it and the character in the same position in the OPTIMAL string. These values 
      are summed and then returned. 
    """

    def cost(self, city1, city2):
        if (city1 == 1 and city2 == 2) or (city1 == 2 and city2 == 1):
            return 10
        elif (city1 == 1 and city2 == 3) or (city1 == 3 and city2 == 1):
            return 20
        elif (city1 == 1 and city2 == 4) or (city1 == 4 and city2 == 1):
            return 23
        elif (city1 == 1 and city2 == 5) or (city1 == 5 and city2 == 1):
            return 53
        elif (city1 == 1 and city2 == 6) or (city1 == 6 and city2 == 1):
            return 12
        elif (city1 == 2 and city2 == 3) or (city1 == 3 and city2 == 2):
            return 4
        elif (city1 == 2 and city2 == 4) or (city1 == 4 and city2 == 2):
            return 15
        elif (city1 == 2 and city2 == 5) or (city1 == 5 and city2 == 2):
            return 32
        elif (city1 == 2 and city2 == 6) or (city1 == 6 and city2 == 2):
            return 17
        elif (city1 == 3 and city2 == 4) or (city1 == 4 and city2 == 3):
            return 11
        elif (city1 == 3 and city2 == 5) or (city1 == 5 and city2 == 3):
            return 18
        elif (city1 == 3 and city2 == 6) or (city1 == 6 and city2 == 3):
            return 21
        elif (city1 == 4 and city2 == 5) or (city1 == 5 and city2 == 4):
            return 9
        elif (city1 == 4 and city2 == 6) or (city1 == 6 and city2 == 4):
            return 5
        else:
            return 15

    # complete fitness function
    def fitness(self, dna):
        fitness=0
        #print(dna)
        for i in range(0,self.DNA_SIZE-1):
            fitness+=self.cost(dna[i],dna[i+1])
        fitness+=self.cost(dna[0],dna[self.DNA_SIZE-1])
        return fitness
    """ 
       For each gene in the DNA, there is a 1/mutation_chance chance that it will be 
       switched out with a random character. This ensures diversity in the 
       population, and ensures that is difficult to get stuck in local minima. 
       """

    def mutate(self, dna, random1, random2):
        if random1 <= 0.01:
            split_pos = int(random2*self.DNA_SIZE)
            half_reversed_dna = dna[split_pos:]
            half_reversed_dna.reverse()
            return dna[:split_pos]+half_reversed_dna
        else:
            return dna

        """
       Slices both dna1 and dna2 into two parts at a random index within their 
       length and merges them. Both keep their initial sublist up to the crossover 
       index, but their ends are swapped. 
       """

    def crossover(self, dna1, dna2, random1, random2):
        half_dna1=[]
        half_dna2=[]
        if random1 <= 0.9:
            splitpos = int(random2 * self.DNA_SIZE)
            cross_dna1 = dna1[splitpos:]
            cross_dna2 = dna2[splitpos:]
            for i in cross_dna1:
                if i not in cross_dna2:
                    half_dna1.append(i)
            for i in cross_dna2:
                if i in cross_dna1:
                    half_dna1.append(i)

            for i in cross_dna2:
                if i not in cross_dna1:
                    half_dna2.append(i)
            for i in cross_dna1:
                if i in cross_dna2:
                    half_dna2.append(i)
            return dna1[:splitpos]+half_dna1,dna2[:splitpos]+half_dna2
        else:
            return dna1,dna2



# endregion
#################################### Algorithms Main Functions #####################################
# region Search_Algorithms_Main_Fn
def SearchAlgorithm_Main():
    searchAlgo = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.',
                                  [0, 15, 2, 100, 60, 35, 30, 3
                                          , 100, 2, 15, 60, 100, 30, 2
                                          , 100, 2, 2, 2, 40, 30, 2, 2
                                          , 100, 100, 3, 15, 30, 100, 2
                                          , 100, 0, 2, 100, 30])
    fullPath, path, TotalCost = searchAlgo.AstarManhattanHeuristic()
    print('**ASTAR with Manhattan Heuristic ** Full Path:' + str(fullPath) + '\nPath is: ' + str(path)
          + '\nTotal Cost: ' + str(TotalCost) + '\n\n')


# endregion

# region KNN_MAIN_FN
'''The dataset classifies tumors into two categories (malignant and benign) (i.e. malignant = 0 and benign = 1)
    contains something like 30 features.
'''


def KNN_Main():
    BC = load_breast_cancer()
    X = []

    for index, row in pd.DataFrame(BC.data, columns=BC.feature_names).iterrows():
        temp = []
        temp.append(row['mean area'])
        temp.append(row['mean compactness'])
        X.append(temp)
    y = pd.Categorical.from_codes(BC.target, BC.target_names)
    y = pd.get_dummies(y, drop_first=True)
    YTemp = []
    for index, row in y.iterrows():
        YTemp.append(row[1])
    y = YTemp;
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1024)
    KNN = KNN_Algorithm(7);
    accuracy = KNN.KNN(X_train, X_test, y_train, y_test)
    print("KNN Accuracy: " + str(accuracy))


# endregion

# region Genetic_Algorithm_Main_Fn
def GeneticAlgorithm_Main():
    genetic = GeneticAlgorithm();
    population = genetic.random_population()
    for generation in range(genetic.GENERATIONS):
        #print("Generation %s... Random sample: '%s'" % (generation, population[0]))
        weighted_population = []

        for individual in population:
            fitness_val = genetic.fitness(individual)
            pair = (individual, 1.0 / fitness_val)
            weighted_population.append(pair)
        population = []

        for _ in range(int(genetic.POP_SIZE / 2)):
            ind1 = genetic.weighted_choice(weighted_population)
            ind2 = genetic.weighted_choice(weighted_population)
            ind1, ind2 = genetic.crossover(ind1, ind2, r.random(),r.random())
            population.append(genetic.mutate(ind1,r.random(),r.random()))
            population.append(genetic.mutate(ind2,r.random(),r.random()))

    fittest_string = population[0]
    minimum_fitness = genetic.fitness(population[0])
    for individual in population:
        ind_fitness = genetic.fitness(individual)
    if ind_fitness <= minimum_fitness:
        fittest_string = individual
        minimum_fitness = ind_fitness

    print(fittest_string)
    print(genetic.fitness(fittest_string))


# endregion
######################## MAIN ###########################33
if __name__ == '__main__':

     SearchAlgorithm_Main()
     KNN_Main()
     GeneticAlgorithm_Main()
