
e=0.5
x=1.0
o=0.0

class TicTacToe:
    def __init__(self):
        self.board = [
            [e,e,e],
            [e,e,e],
            [e,e,e]
            ]
        self.turn =x

    def printBoard(self):
        print("-------------")
        for row in self.board:
            print("| ", end="")
            for symbol in row:
                print("X" if symbol ==x else ("O" if symbol ==  o else " "), end=" | ")
            print("")
            print("-------------")
        #print((("X" if symbol ==x else "O") for symbol in row + "\n") for row in self.board)
    def toArray(self):
        return self.board[0] + self.board[1] + self.board[2]
    
    def loadArray(self,array, turn):
        self.board = [array[0:3],array[3:6],array[6:9]]
        self.turn = turn


    def takeTurn(self, row, col):
        if(self.board[row][col] !=  e):
            return -1
        self.board[row][col] = self.turn
        if self.turn ==x:
            self.turn =  o
            return x
        else:
            self.turn =x
            return o

        
    def checkWin(self):
        ##row wins

        for row in self.board:
            owin = True
            xwin = True
            for i in range(3):
                if row[i] !=x:
                    xwin = False
                if row[i] !=  o:
                    owin = False
            if owin:
                return  o
            if xwin: 
                return x

        ##col wins
        for i in range (3):
            owin = True
            xwin = True
            for j in range (3):
                if self.board[j][i] !=x:
                    xwin = False
                if self.board[j][i] !=  o:
                    owin = False
            if owin:
                return  o
            if xwin: 
                return x

        ##diag wins
        xwin = True
        owin = True
        for i in range(3):
            if self.board[i][i] != x:
                xwin = False
            if self.board[i][i] != o:
                owin = False
        if owin:
            return  o
        if xwin: 
            return x

        xwin = True
        owin = True
        for i in range(3):
            if self.board[i][2-i] != x:
                xwin = False
            if self.board[i][2-i] != o:
                owin = False
        if owin:
            return  o
        if xwin: 
            return x
        ##draw
        draw = True
        for i in range (3):
            for j in range (3):
                if self.board[i][j] ==  e:
                    draw = False
        if draw:
            return 0.5
        return -1