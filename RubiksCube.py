import random
import sys
from time import time
import pickle
from collections import deque, OrderedDict
import os.path

# different moves
# https://ruwix.com/online-puzzle-simulators/2x2x2-pocket-cube-simulator.php

MOVES = {
    "U": [2,  0,  3,  1, 20, 21,  6,  7,  4,  5, 10, 11, 12, 13, 14, 15,  8,  9, 18, 19, 16, 17, 22, 23],
    "U'": [1,  3,  0,  2,  8,  9,  6,  7, 16, 17, 10, 11, 12, 13, 14, 15, 20, 21, 18, 19,  4,  5, 22, 23],
    "R": [0,  9,  2, 11,  6,  4,  7,  5,  8, 13, 10, 15, 12, 22, 14, 20, 16, 17, 18, 19,  3, 21,  1, 23],
    "R'": [0, 22,  2, 20,  5,  7,  4,  6,  8,  1, 10,  3, 12, 9, 14, 11, 16, 17, 18, 19, 15, 21, 13, 23],
    "F": [0,  1, 19, 17,  2,  5,  3,  7, 10,  8, 11,  9, 6,  4, 14, 15, 16, 12, 18, 13, 20, 21, 22, 23],
    "F'": [0,  1,  4,  6, 13,  5, 12,  7,  9, 11,  8, 10, 17, 19, 14, 15, 16,  3, 18,  2, 20, 21, 22, 23],
    "D": [0,  1,  2,  3,  4,  5, 10, 11,  8,  9, 18, 19, 14, 12, 15, 13, 16, 17, 22, 23, 20, 21,  6,  7],
    "D'": [0,  1,  2,  3,  4,  5, 22, 23,  8,  9,  6,  7, 13, 15, 12, 14, 16, 17, 10, 11, 20, 21, 18, 19],
    "L": [23,  1, 21,  3,  4,  5,  6,  7,  0,  9,  2, 11, 8, 13, 10, 15, 18, 16, 19, 17, 20, 14, 22, 12],
    "L'": [8,  1, 10,  3,  4,  5,  6,  7, 12,  9, 14, 11, 23, 13, 21, 15, 17, 19, 16, 18, 20,  2, 22,  0],
    "B": [5,  7,  2,  3,  4, 15,  6, 14,  8,  9, 10, 11, 12, 13, 16, 18,  1, 17,  0, 19, 22, 20, 23, 21],
    "B'": [18, 16,  2,  3,  4,  0,  6,  1,  8,  9, 10, 11, 12, 13,  7,  5, 14, 17, 15, 19, 21, 23, 20, 22],
}

MOVES_LIST = ["U", "U'", "R", "R'", "F", "F'", "D", "D'", "L", "L'", "B", "B'"] #[::-1]

returnOpposite = {"R": "O", "G": "B", "B": "G", "O": "R", "W": "Y", "Y": "W"}

oppositeMoves = {
    "F": "F'",
    "B": "B'",
    "L": "L'",
    "R": "R'",
    "U": "U'",
    "D": "D'",
    "F'": "F",
    "B'": "B",
    "L'": "L",
    "R'": "R",
    "U'": "U",
    "D'": "D",
}

complements = {
    "U": "D'",
    "D'": "U",
    "D": "U'",
    "U'": "D",
    "L": "R'",
    "R'": "L",
    "L'": "R",
    "R": "L'",
    "F": "B'",
    "B'": "F",
    "F'": "B",
    "B": "F'",
}

corners = [[8, 17, 2], [9, 4, 3], [10, 19, 12], [ 11, 6, 13], [20, 5, 1], [22, 7, 15], [23, 18, 14], [21, 16, 0]]

cornersIdeal = [["G", "O", "W"], ["G", "R", "W"], ["G", "O", "Y"], ["G","R", "Y"], ["B", "R", "W"], ["B", "R", "Y"], ["B", "O", "Y"], ["B", "O", "W"]]

cornersCoord = [[0,0,0], [1,0,0], [0,1,0], [1,1,0], [1,0,1], [1,1,1], [0,1,1], [0,0,1]]

cornersIdealSorted = [sorted(x) for x in cornersIdeal]

stringToInt = {
    "W": 1,
    "R": 2,
    "G": 3,
    "O": 4,
    "Y": 5,
    "B":6
}

intToString = {
    1: "W",
    2: "R",
    3: "G",
    4: "O",
    5: "Y",
    6:"B"
}

"""
sticker indices:

      0  1               
      2  3
16 17  8  9   4  5  20 21
18 19  10 11  6  7  22 23
      12 13
      14 15

face colors:

    0
  4 2 1 5
    3

moves:
[ U , U', R , R', F , F', D , D', L , L', B , B']
"""


class node:
    def __init__(self, state, parent, value):
        self.parent = parent
        self.state = state
        self.val = value
        if self.parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

        if self.parent is None:
            self.path = ""
        else:
            self.path = self.parent.path + " " + self.val

    def printTree(self):

        sel = self

        par = self.parent if self.parent is not None else None

        gpar = par.parent if par is not None and par.parent is not None else None

        if gpar is not None and gpar.parent is not None:
            gpar.parent.printTree() 

        if par is None:
            printThree(sel.state.string)
        elif gpar is None:
            printThree(par.state.string, sel.state.string)
        else:
            printThree(gpar.state.string, par.state.string, sel.state.string)
        


        print("")
        return
    
    def __repr__(self):
        return str(self.val)

    def heuristic(self):
        cornersCurr = []

        for x in corners:
            cornersCurr.append([self.state.string[x[0]], self.state.string[x[1]], self.state.string[x[2]]])
        
        [x.sort() for x in cornersCurr]

        diff = 0

        for i in range(len(cornersIdealSorted)):

            diff_x = abs(cornersCoord[cornersIdealSorted.index(cornersCurr[i])][0] - cornersCoord[i][0])
            diff_y = abs(cornersCoord[cornersIdealSorted.index(cornersCurr[i])][1] - cornersCoord[i][1])
            diff_z = abs(cornersCoord[cornersIdealSorted.index(cornersCurr[i])][2] - cornersCoord[i][2])

            diff += diff_x+diff_y+diff_z      
        return self.depth + (diff / 4)
   
    def heuristicBetter(self, db):
        return db[mapBackandForth(self.state.norm().string)]

class cube:
    def __init__(self, string="WWWW RRRR GGGG YYYY OOOO BBBB"):
        # normalize stickers relative to a fixed corner
        self.string = string.replace(" ", "")
        self.norm()
        return

    def norm(self):

        global returnOpposite

        map = {
            self.string[10]: "G",
            self.string[12]: "Y",
            self.string[19]: "O",
            returnOpposite[self.string[10]]: "B",
            returnOpposite[self.string[12]]: "W",
            returnOpposite[self.string[19]]: "R",
        }

        newString = []
        try:
            for char in self.string:
                newString.append(map[char])
        except Exception:
            return self

        self.string = "".join(newString)
        return self

    def equals(self, cube):
        return self.norm().string == cube.norm().string

    def clone(self):
        return cube(self.string)

    def applyMove(self, move):
        newState = []

        for index in MOVES[move]:
            newState.append(self.string[index])

        self.string = "".join(newState)

        return self

        # apply a string sequence of moves to a state

    def applyMovesStr(self, alg):

        clonedState = self.clone()

        for move in alg.split():
            clonedState.applyMove(move)

        return clonedState

    def isSolved(self):
        return self.norm().string == "WWWWRRRRGGGGYYYYOOOOBBBB"

    def print(self):
        arrayOfString = []

        for char in self.string:
            arrayOfString.append(char)

        idealRepr = [
            [" ", " ", " ", 0, 1, " ", " ", " ", " "],
            [" ", " ", " ", 2, 3, " ", " ", " ", " "],
            [16, 17, " ", 8, 9, " ", 4, 5, " ", 20, 21],
            [18, 19, " ", 10, 11, " ", 6, 7, " ", 22, 23],
            [" ", " ", " ", 12, 13, " ", " ", " ", " ", " "],
            [" ", " ", " ", 14, 15, " ", " ", " ", " ", " "],
        ]

        for row in idealRepr:
            for col in row:
                if col == " ":
                    print(" ", end="")
                else:
                    print(arrayOfString[col], end="")
            print("")

        return

    def shuffle(self, n):

        global MOVES_LIST
        randomMoves = random.choices(MOVES_LIST, k=int(n))
        print(" ".join(randomMoves))
        newStateCube = self.applyMovesStr(" ".join(randomMoves))

        newStateCube.print()

        return self

    def randomWalk(self, n=3):
        global MOVES_LIST
        temporaryCopyOfCube = self.clone()
        while not temporaryCopyOfCube.isSolved():
            randomMoves = random.choices(MOVES_LIST, k=int(n))
            movesApplied = []
            temporaryCopyOfCube = self.clone()
            for move in randomMoves:
                if not temporaryCopyOfCube.isSolved():
                    temporaryCopyOfCube.applyMove(move)
                    movesApplied.append(move)
        print(" ".join(movesApplied))
        temporaryCopyOfCube.print()

    def moveSanityChecker(self, currMove):
        prevMove = currMove.parent
        last4moves = [currMove.val]
        temp = currMove.parent
        for _ in range(3):
            if temp.parent is not None:
                last4moves.append(temp.val)
                temp = temp.parent

        return not (
            oppositeMoves[currMove.val] == prevMove.val
            or complements[currMove.val] == prevMove.val
            or (len(last4moves) >= 3 and len(set(last4moves)) == 1)
        )

    def bfs(self):
        OPEN, CLOSED, start, i, currentNode = (
            deque([node(self.clone(), None, None)]),
            deque(),
            time(),
            0,
            None,
        )

        openAndClosed = {}
        openAndClosed[self.norm().string] = ""

        try:

            while len(OPEN) != 0:
                currentNode = OPEN.popleft()

                if currentNode.state.isSolved():
                    break

                CLOSED.append(currentNode)
                tempString = currentNode.state.norm().string
                if tempString not in openAndClosed:
                    openAndClosed[tempString] = ""
                i = i + 1

                for mv in MOVES_LIST:
                    tempState = currentNode.state.applyMovesStr(mv).norm()
                    if (tempState.string not in openAndClosed):
                        tempNode = node(tempState, currentNode, mv)
                        if self.moveSanityChecker(tempNode):
                            OPEN.append(tempNode)
                            openAndClosed[tempState.string] = ""
                
        except:
            print("\niterations:" + str(i))
            print("%.2f seconds" % (time() - start))
            return            

        print(currentNode.path)
        currentNode.printTree()
        print("iterations:" + str(i))
        print("%.3f seconds" % (time() - start))
    
    def dfs(self, maxDepth):
        OPEN, CLOSED, i, currentNode = (
            OrderedDict(),
            OrderedDict(),
            0,
            None,
        )
        OPEN[node(self.clone(), None, None)] = ""

        while len(OPEN) != 0:

            currentNode , _ = OPEN.popitem(True)

            if currentNode.state.isSolved():
                break

            CLOSED[currentNode] = ""

            if currentNode.depth < maxDepth:
                for mv in MOVES_LIST:                     
                    tempNode = node(currentNode.state.applyMovesStr(mv), currentNode, mv)
                    if self.moveSanityChecker(tempNode):
                        i = i + 1
                        OPEN[tempNode] = ""


        return currentNode, i
    
    def ids(self, moves, maxDepth):
        start = time()
        i = 0
        for x in range(maxDepth+1):
            copy = self.applyMovesStr(moves)
            node, d = copy.dfs(x)
            i += d
            print("Depth: %d d: %d" % (x, d))
            if (node.state.isSolved()):
                print("IDS found a solution at depth %d" % x)
                print(node.path)
                node.printTree()
                break
        print("iterations: %d" % i)
        print("%.3f seconds" % (time() - start))

    def astar(self):
        OPEN, CLOSED, start, i, currentNode = (
            [],
            [],
            time(),
            0,
            None,
        )

        startNode = node(self.clone(), None, None)

        OPEN.append((startNode.heuristic(), startNode))        

        openAndClosed = {}
        openAndClosed[self.norm().string] = ""


        while len(OPEN) != 0:
            OPEN.sort(key=lambda x: x[0])
            _ , currentNode = OPEN.pop(0)

            if currentNode.state.isSolved():
                break

            CLOSED.append((currentNode.heuristic(), currentNode))
            tempString = currentNode.state.norm().string
            if tempString not in openAndClosed:
                openAndClosed[tempString] = ""
            i = i + 1

            for mv in MOVES_LIST:
                tempState = currentNode.state.applyMovesStr(mv).norm()
                if (tempState.string not in openAndClosed):
                    tempNode = node(tempState, currentNode, mv)
                    if self.moveSanityChecker(tempNode):
                        OPEN.append((tempNode.heuristic() , tempNode))
                        openAndClosed[tempState.string] = ""        



        print(currentNode.path.strip() + "\n")
        currentNode.printTree()
        print("iterations:" + str(i))
        print("%.3f seconds" % (time() - start))    
    
    def astarBetter(self, db):
        OPEN, CLOSED, start, i, currentNode = (
            [],
            [],
            time(),
            0,
            None,
        )

        startNode = node(self.clone(), None, None)

        OPEN.append((startNode.heuristicBetter(db), startNode))        

        openAndClosed = {}
        openAndClosed[self.norm().string] = ""


        while len(OPEN) != 0:
            OPEN.sort(key=lambda x: x[0])
            _ , currentNode = OPEN.pop(0)

            if currentNode.state.isSolved():
                break

            CLOSED.append((currentNode.heuristicBetter(db), currentNode))
            tempString = currentNode.state.norm().string
            if tempString not in openAndClosed:
                openAndClosed[tempString] = ""
            i = i + 1

            for mv in MOVES_LIST:
                tempState = currentNode.state.applyMovesStr(mv).norm()
                if (tempState.string not in openAndClosed):
                    tempNode = node(tempState, currentNode, mv)
                    if self.moveSanityChecker(tempNode):
                        OPEN.append((tempNode.heuristicBetter(db) , tempNode))
                        openAndClosed[tempState.string] = ""

        print(currentNode.path.strip() + "\n")
        currentNode.printTree()
        print("iterations:" + str(i))
        print("%.3f seconds" % (time() - start))

    def dlastar(self, limit):
        OPEN, CLOSED, start, i, currentNode = (
            [],
            [],
            time(),
            0,
            None,
        )

        startNode = node(self.clone(), None, None)

        OPEN.append((startNode.heuristic(), startNode))        

        openAndClosed = {}
        openAndClosed[self.norm().string] = ""


        while len(OPEN) != 0:
            #OPEN.sort(key=lambda x: x[0])
            _ , currentNode = OPEN.pop(0)

            if currentNode.state.isSolved():
                break

            tempH = currentNode.heuristic()
            CLOSED.append((tempH, currentNode))
            #tempString = currentNode.state.norm().string
            #if tempString not in openAndClosed:
            #    openAndClosed[tempString] = ""
            i = i + 1

            if tempH <= limit: 
                for mv in MOVES_LIST:
                        tempNode = node(currentNode.state.applyMovesStr(mv).norm(), currentNode, mv)
                        if self.moveSanityChecker(tempNode):
                            OPEN.append((tempNode.heuristic(), tempNode))
                            openAndClosed[tempNode.state.string] = ""


        return currentNode, i

    def ida(self, moves, limit):
        start = time()
        a = cube()
        n = node(a.applyMovesStr(moves), None, None)
        i = n.heuristic()
        iterations = 0
        while i <= limit:
            copy = self.applyMovesStr(moves)
            Node, d = copy.dlastar(i)
            iterations += d
            print("H limit: %f d: %d" % (i, d))
            if (Node.state.isSolved()):
                print("IDA found a solution at H %f" % i)
                print(Node.path)
                Node.printTree()
                break
            i += 0.5
        print("H: %f" % i)
        print("iterations: %d" % iterations)
        print("%.3f seconds" % (time() - start))

def mapBackandForth(n):
    if isinstance(n, str):
        temp = list(map(lambda x: str(stringToInt[x]), n))
        return int("".join(temp))
    elif isinstance(n, int):
        n = str(n)
        temp = list(map(lambda x: str(intToString[int(x)]), n))
        return "".join(temp)
      
def create_pattern_database():
    temp = cube()
    tempNode = node(temp.clone(), None, None)
    OPEN, CLOSED, start, i, currentNode = (
            deque([tempNode]),
            deque(),
            time(),
            0,
            None,
        )

    openAndClosed = {}
    openAndClosed[temp.norm().string] = ""

    table = {}
    table[mapBackandForth(temp.string)] = tempNode.depth

    
    with open("log.txt", 'w+') as f:
        while len(OPEN) != 0 and len(openAndClosed) < 3674160:
            currentNode = OPEN.popleft()
            CLOSED.append(currentNode)
            tempString = currentNode.state.norm().string
            if tempString not in openAndClosed:
                openAndClosed[tempString] = ""
            i = i + 1

            for mv in MOVES_LIST:
                tempState = currentNode.state.applyMovesStr(mv).norm()
                if (tempState.string not in openAndClosed):
                    tempNode = node(tempState, currentNode, mv)
                    table[mapBackandForth(tempState.string)] = tempNode.depth
                    if cube().moveSanityChecker(tempNode):
                        OPEN.append(tempNode)
                        openAndClosed[tempState.string] = ""
            
            if (len(table.keys()) % 100 == 0):
                    f.seek(0)
                    f.write("Progress: %d/3674160 nodes" % len(table.keys()))
                    f.truncate()

    with open(r'pdb.pkl', 'wb') as f:
        pickle.dump(table, f)

    print("iterations:" + str(i))
    print("%.3f seconds" % (time() - start))
    print("database created!")

def printThree(x, y = None, z = None):
        arrayOfStringX = []
        for char in x:
            arrayOfStringX.append(char)

        arrayOfStringY = []
        if y is not None:
            for char in y:
                arrayOfStringY.append(char)

        arrayOfStringZ = []
        if z is not None:
            for char in z:
                arrayOfStringZ.append(char)

        idealRepr = [
            [" ", " ", " ", 0, 1, " ", " ", " ", " "],
            [" ", " ", " ", 2, 3, " ", " ", " ", " "],
            [16, 17, " ", 8, 9, " ", 4, 5, " ", 20, 21],
            [18, 19, " ", 10, 11, " ", 6, 7, " ", 22, 23],
            [" ", " ", " ", 12, 13, " ", " ", " ", " ", " "],
            [" ", " ", " ", 14, 15, " ", " ", " ", " ", " "],
        ]

        for row in idealRepr:
            for col in row:
                if col == " ":
                    print(" ", end="")
                else:
                    print(arrayOfStringX[col], end="")
            print("\t", end="")
            for col in row:
                if col == " " and y is not None:
                    print(" ", end="")
                elif y is not None:
                    print(arrayOfStringY[col], end="")
            print("\t", end="")
            for col in row:
                if col == " " and z is not None:
                    print(" ", end="")
                elif z is not None:
                    print(arrayOfStringZ[col], end="")
            print("")

        return

if __name__ == "__main__":   
    
    if len(sys.argv) == 2:
        myCube = cube()
        if sys.argv[1] == "print":
            myCube.print()
        elif sys.argv[1] == "goal":
            print(myCube.isSolved())
        elif sys.argv[1] == "norm":
            myCube.norm()
        elif sys.argv[1] == "createDB":
            create_pattern_database()
        else:
            print("Not a valid command, sir/ma'am. Please try again.")
    elif len(sys.argv) == 3:
        if sys.argv[1] == "print":
            myCube = cube(sys.argv[2])
            myCube.print()
        elif sys.argv[1] == "goal":
            myCube = cube(sys.argv[2])
            print(myCube.isSolved())
        elif sys.argv[1] == "applyMovesStr":
            myCube = cube()
            myCube.applyMovesStr(sys.argv[2]).norm().print()
        elif sys.argv[1] == "norm":
            myCube = cube(sys.argv[2])
            myCube.norm().print()
        elif sys.argv[1] == "shuffle":
            myCube = cube()
            myCube.shuffle(sys.argv[2])
        elif sys.argv[1] == "random":
            myCube = cube()
            newCubeState = myCube.applyMovesStr(sys.argv[2])
            newCubeState.randomWalk()
        elif sys.argv[1] == "bfs":
            myCube = cube()
            myCube.applyMovesStr(sys.argv[2]).bfs()
        elif sys.argv[1] == "ids":
            myCube = cube()
            myCube.ids(sys.argv[2], 1000)
        elif sys.argv[1] == "idastar":
            myCube = cube()
            myCube.ida(sys.argv[2], 1000)
        elif sys.argv[1] == "astarMD":
            myCube = cube()
            myCube.applyMovesStr(sys.argv[2]).astar()
        elif sys.argv[1] == "astar":
            myCube = cube()
            if not os.path.isfile("pdb.pkl"):
                print("Database does not exist, need to create one.")
                print("Check log.txt for progress.")
                print("Not sure why database? Please refer to my documentation for explanation.")
                create_pattern_database()
            print("Loading pattern database, please hold....")
            with open(r'pdb.pkl', 'rb') as f:
                db = pickle.load(f)
            print("Database loaded!! Commencing request!")
            myCube.applyMovesStr(sys.argv[2]).astarBetter(db)
        else:
            print("Not a valid command, sir/ma'am. Please try again.")
    elif len(sys.argv) == 4:
        myCube = cube(sys.argv[3])
        if sys.argv[1] == "applyMovesStr":
            myCube.applyMovesStr(sys.argv[2]).print()
        else:
            print("Not a valid command, sir/ma'am. Please try again.")
