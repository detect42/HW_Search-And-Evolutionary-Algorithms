# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from game import Agent
from game import Actions
import copy
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    startNode = problem.getStartState()

    if problem.isGoalState(startNode):
        return []

    Stack = util.Stack()
    VisitedNode = []
    Stack.push((startNode, []))

    while not Stack.isEmpty():
        Nownode, actions = Stack.pop()
        if (Nownode not in VisitedNode):
            VisitedNode.append(Nownode)
            if problem.isGoalState(Nownode):
                return actions
            for nextNode, nextAction, cost in problem.getSuccessors(Nownode):
                newAction = actions + [nextAction]
                Stack.push((nextNode, newAction))

    util.raiseNotDefined()


class Stack:
    pass


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    startNode = problem.getStartState()

    if problem.isGoalState(startNode):
        return []

    Que= util.Queue()
    VisitedNode = []
    Que.push((startNode, []))

    while not Que.isEmpty():
        Nownode, actions = Que.pop()
        if (Nownode not in VisitedNode):
            VisitedNode.append(Nownode)
            if problem.isGoalState(Nownode):
                return actions
            for nextNode, nextAction, cost in problem.getSuccessors(Nownode):
                newAction = actions + [nextAction]
                Que.push((nextNode, newAction))

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """




    return 0

def myHeuristic(state, map, problem=None):
    """
        you may need code other Heuristic function to replace  NullHeuristic
        """
    (Goal_x, Goal_y) = problem.goal

    map = [[0 for i in range(problem.walls.height)] for j in range(problem.walls.width)]
    for i in range(problem.walls.width):
        for j in range(problem.walls.height):
            if problem.walls[i][j]:
                continue
            # print(i,j)
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                x, y = i, j
                dx, dy = Actions.directionToVector(action)
                nextx, nexty = int(x + dx), int(y + dy)
                if nextx < 0 or nexty < 0 or nextx >= problem.walls.width or nexty >= problem.walls.height:
                    map[i][j] += 1
                elif problem.walls[nextx][nexty]:
                    map[i][j] += 1
    for i in range(problem.walls.width):
        for j in range(problem.walls.height):
            if map[i][j] == 3:
                map[i][j] = 100
            elif map[i][j] == 2:
                map[i][j] = 0
            elif map[i][j] == 1:
                map[i][j] = -1
            elif map[i][j] == 0:
                map[i][j] = -5
    map[1][1] = 0

    #print(state[0], state[1], (abs(state[0] - Goal_x)**2 + abs(state[1] - Goal_y)**2)**0.5  + map[state[0]][state[1]])
   # return 0
    return (abs(state[0] - Goal_x)**2 + abs(state[1] - Goal_y)**2)**0.5  + map[state[0]][state[1]]
    #return (abs(state[0] - Goal_x) + abs(state[1] - Goal_y))  + map[state[0]][state[1]]
   # return (abs(state[0] - Goal_x) + abs(state[1] - Goal_y))*10  + map[state[0]][state[1]]

def Dij(S,T,problem):

    P_Que = util.PriorityQueue()
    VisitedNode = []
  #  print("S= {}".format(S))
    # item: (node, action, cost)
    P_Que.push((S,0) , 0)
    while not P_Que.isEmpty():
        Now = P_Que.pop()
        cur, dis = Now[0], Now[1]
     #   print(cur,"##",dis)
        if cur not in VisitedNode:
            VisitedNode.append(cur)
            if cur == T: return dis
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                x, y = cur[0],cur[1]
                dx, dy = Actions.directionToVector(action)
                nextx, nexty = int(x + dx), int(y + dy)
                if nextx < 0 or nexty < 0 or nextx >= problem.walls.width or nexty >= problem.walls.height or problem.walls[nextx][nexty] or VisitedNode.__contains__((nextx,nexty)):
                    continue
                else:
                    P_Que.push(((nextx,nexty),dis+1),dis+1)

def Build(problem):
    A = C = problem.walls.width
    B = D = problem.walls.height
    Map = [[[[(0) for _ in range(D)] for _ in range(C)] for _ in range(B)] for _ in range(A)]

    for i in range(problem.walls.width):
        for j in range(problem.walls.height):
            for k in range(problem.walls.width):
                for l in range(problem.walls.height):
                    if problem.walls[i][j] or problem.walls[k][l]:
                        continue
                    Map[i][j][k][l] = Dij((i,j),(k,l),problem)
    return Map
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first.

        Your search algorithm needs to return a list of actions that reaches the
        goal. Make sure to implement a graph search algorithm.

        To get started, you might want to try some of these simple commands to
        understand the search problem that is being passed in:


        print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
        print("Start's successors:", problem.getSuccessors(problem.getStartState()))
        """
    print("Start:", problem.getStartState())


    print(problem.walls.width , problem.walls.height)
    Map = Build(problem)
    startNode = problem.getStartState()
    startcost = heuristic(startNode, problem , Map)
    if problem.isGoalState(startNode):
        return []

    P_Que = util.PriorityQueue()
    VisitedNode = []

    print(startcost)

    # item: (node, action, cost)
    P_Que.push((startNode, [], 0), startcost)
    Show=0
    while not P_Que.isEmpty():
        (currentNode, actions, preCost) = P_Que.pop()
       # print(currentNode[0],preCost+heuristic(currentNode, problem,Map))
        if Show < preCost + heuristic(currentNode, problem,Map):
            Show = preCost + heuristic(currentNode, problem,Map)
            print("noedepth: {0}".format(Show))
        if (currentNode not in VisitedNode):
            VisitedNode.append(currentNode)

            if problem.isGoalState(currentNode):
                return actions

            for nextNode, nextAction, nextCost in problem.getSuccessors(currentNode):
                newAction = actions + [nextAction]
                G_Cost = problem.getCostOfActions(newAction)
                newPriority = G_Cost + heuristic(nextNode, problem,Map)
                P_Que.push((nextNode, newAction, G_Cost), newPriority)

    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
