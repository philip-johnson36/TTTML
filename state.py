from tictactoe import TicTacToe


x=1.0
o=0.0
e=0.5

class State:
    def __init__(self, boardState):
        self.state = boardState.copy()
        self.games = 0
        self.xwins = 0

        self.children = []
        for i in range(9):
            if self.state[i]==e:
                copy = self.state.copy()
                copy[i] = self.getTurn()
                self.children.append(copy)
            else:
                self.children.append(None)
    def getTurn(self):
        if sum(self.state)==4.5:return x
        elif sum(self.state)==5:return o
        return -1
    def getIndex(self):
        index=0
        for i in range(len(self.state)):
            index+= pow(3,i) * 2 * self.state[i]
        return index
    def getEval(self):
        if self.games==0:return -1.0
        return self.xwins/self.games
    def getAllChildren(self):
        return self.children
    def hasValidChildEvals(self):
        if self.getTurn()==-1:return False
        game = TicTacToe()
        game.loadArray(self.state, self.getTurn())
        if game.checkWin()!=-1:return False
        return True

