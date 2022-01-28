import itertools
import math
import time
from torch import nn
from numpy import Infinity
from state import State
from neuralnetwork import NeuralNetwork
from tictactoe import TicTacToe
import torch
import random
from tictacdata import TicTacDataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import numpy as np

x=1.0
e=0.5
o=0.0
emptyState = [e,e,e,e,e,e,e,e,e]

class Trainer:
    def __init__(self):


        self.model = NeuralNetwork()
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
    
    def optimize(self):
        self.generateDataset(True)

        while True:
            im = input("Options: \n's [x]' to train for x epochs\n'd' to regenerate dataset\n'r' to have the AI play a match vs a random player\n'p' to play vs the AI\n'l' to load a trained network\n>>>")
            if(len(im)==0):continue
            if(im[0]=="s"):
                if(len(im)==1):im=1
                else:im=im.split()[1]
                im = int(im)
                for i in range(im):
                    print(f"Epoch {i+1}/{im}")
                    self.train_loop()
                    #self.test_loop()
            elif(im=="d"):
                self.generateDataset()
            elif(im=="r"):
                print(f"Bot scored {100*self.randomMatchDet(5000,x):.2f}% playing as x")
                print(f"Bot scored {100*self.randomMatchDet(5000,o):.2f}% playing as o")
            elif(im=="p"):
                self.playHuman()
            elif(im=="w"):
                torch.save(self.model.state_dict(), 'trained.pth')
            elif(im=="l"):
                self.model.load_state_dict(torch.load('trained.pth'))
                self.model.eval()



    def train_loop(self):
        size = len(self.trainDataloader.dataset)
        lossSum = 0
        for batch, (X, y) in enumerate(self.trainDataloader):
            # Compute prediction and loss
            
            pred = self.model(X)
            pred=torch.reshape(pred, y.size())

            loss = self.loss(pred, y)
            lossSum += loss.item()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        lossSum /= len(self.trainDataloader)
        print(f"Training loss: {lossSum*100:>6f}")

    def test_loop(self):
        size = len(self.testDataloader.dataset)
        num_batches = len(self.testDataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in self.testDataloader:
                pred = self.model(X)
                pred=torch.reshape(pred, y.size())

                test_loss += self.loss(pred, y).item()
                correct += (pred.argmax(0) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test loss: {test_loss*100:>6f} \n")
    def generateDataset(self, rando=False):
        self.posToEval = [{}for _ in range(len(emptyState)+1)]
        while sum([len(self.posToEval[i])for i in range(len(emptyState)+1)])<4000:
            game = TicTacToe()
            gamestates = [State([e,e,e,e,e,e,e,e,e])]
            while game.checkWin()==-1:
                while True:
                    choice = random.randint(0,len(emptyState)-1)
                    if(game.takeTurn(math.floor(choice/3), choice%3)!=-1):break
                gamestates.append(State(game.toArray()))
            for i in range(len(gamestates)):
                self.posToEval[i][getIndex(gamestates[i].state)] = gamestates[i]
                
        count=0
        for i in range(len(emptyState)+1):
            for pos in self.posToEval[i]:
                if(count%100==0):print(f"Generating data for position {count}/4000")
                state = self.posToEval[i][pos]
                while state.games<100:
                    self.playGame(state.state, rando)
                count+=1


        self.legalIndexes=[self.posToEval[i].keys() for i in range(len(emptyState)+1)]

        trData = [self.posToEval[z][idx].state for z in range(len(emptyState)+1) for idx in self.legalIndexes[z] ]
        trLabels = [self.posToEval[z][idx].getEval() for z in range(len(emptyState)+1) for idx in self.legalIndexes[z] ]

        
        teData = trData.copy()
        teLabels = trLabels.copy()

        print(trData[0],trLabels[0])

        self.testDataset = TicTacDataset(teLabels, teData)
        self.trainingDataset = TicTacDataset(trLabels, trData)


        self.testDataloader = DataLoader(self.testDataset, batch_size = 64)
        self.trainDataloader = DataLoader(self.trainingDataset, batch_size = 64)


    def playGame(self, startState, rando=False):
        buffer = sum([1 if startState[i]!=e else 0 for i in range(len(emptyState))])
        game = TicTacToe()
        game.loadArray(startState, State(startState).getTurn())
        turnCounter = State(startState).getTurn()
        stateIndexes = [getIndex(startState)]
        while(game.checkWin()==-1):
            if rando:           
                gamestate = game.toArray()
                while True:
                    choice = random.randint(0,len(emptyState)-1)
                    if(game.takeTurn(math.floor(choice/3), choice%3)!=-1):break
            else:
                gamestate = game.toArray()
                children = State(gamestate).getAllChildren()
                evals = {}
                for i in range(len(emptyState)):
                    if(children[i]==None):continue
                    evals[i]=self.model(torch.tensor(children[i]))   
                while(True):
                    indy = torch.multinomial(nn.Softmax(dim=0)(torch.tensor(list(evals.values()))*5),1).item() if turnCounter==x else torch.multinomial(nn.Softmax(dim=0)(torch.tensor([1-list(evals.values())[i]for i in range(len(evals.values()))])*5),1).item()
                       
                    choice= [key for key in evals if list(evals.keys()).index(key) == indy][0]  if turnCounter==x else [key for key in evals if list(evals.keys()).index(key) == indy][0]  
                    if gamestate[choice]==0.5:break
                    evals.pop(choice)
            game.takeTurn(math.floor(choice/3), choice%3)
            gamestate = game.toArray()
            stateIndexes.append(getIndex(gamestate))
            turnCounter = (turnCounter+1)%2
        winner = game.checkWin()
        for i in range(len(stateIndexes)):
            if stateIndexes[i] in self.posToEval[i+buffer].keys():
                self.posToEval[i+buffer][stateIndexes[i]].games+=1
                self.posToEval[i+buffer][stateIndexes[i]].xwins+=winner


    def playHuman(self):
        while True:
            xo = input("Want to play x or o?")
            if(xo=="x"):
                xo = x
                break
            if(xo=="o"):
                xo = o
                break
            print("Invalid input!")
        game = TicTacToe()
        turnCounter = 1 if xo == x else 0
        while(game.checkWin()==-1):
            if(turnCounter%2==0):               
                gamestate = game.toArray()
                children = State(gamestate).getAllChildren()
                evals = {}
                for i in range(len(emptyState)):
                    if(children[i]==None):continue
                    evals[i]=self.model(torch.tensor(children[i]))   
                while(True):                        
                    choice= [key for key in evals if evals[key] == max(evals.values())][0]  if xo==o else [key for key in evals if evals[key] == min(evals.values())][0]  
                    if gamestate[choice]==0.5:break
                    evals.pop(choice)
                game.takeTurn(math.floor(choice/3), choice%3)
            else:
                while True:
                    while True:
                        row = input("Please enter a row: ")
                        if(row=="0" or row=="1" or row=="2"):
                            break
                    while True:
                        col = input("Please enter a column: ")
                        if(col=="0" or col=="1" or col=="2"):
                            break
                    choice = 3*int(row) + int(col)
                    if(game.takeTurn(math.floor(choice/3), choice%3)!=-1):break
                    print("Invalid move!")
            turnCounter+=1
            game.printBoard()
            print(f"The computer evaluates this position as {self.model(torch.tensor(game.toArray())).item():.4f}")
        winner = game.checkWin()
        if(winner==0.5):
            print(f"Game over! It's a draw.")
            return
        print(f"Game over! {'x' if winner == 1 else 'o'} won.")
        

    def randomMatchDet(self, numGames, xo):
        numDubs = 0
        for i in range(numGames):
            game = TicTacToe()
            turnCounter = 0 if xo == x else 1
            while(game.checkWin()==-1):
                if(turnCounter%2==0):               
                    gamestate = game.toArray()
                    children = State(gamestate).getAllChildren()
                    evals = {}
                    for i in range(len(emptyState)):
                        if(children[i]==None):continue
                        evals[i]=self.model(torch.tensor(children[i]))   
                    while(True):                        
                        choice= [key for key in evals if evals[key] == max(evals.values())][0]  if xo==x else [key for key in evals if evals[key] == min(evals.values())][0]  
                        if gamestate[choice]==0.5:break
                        evals.pop(choice)
                else:
                    choice = random.randint(0,len(emptyState)-1)
                game.takeTurn(math.floor(choice/3), choice%3)
                turnCounter+=1
            winner = game.checkWin()
            numDubs += winner if xo == x else 1-winner
        return numDubs/numGames

            
            
def getIndex(gamestate):
    index=0
    for i in range(len(gamestate)):
        index+= pow(3,i) * 2 * gamestate[i]
    return math.floor(index)
def getGamestate(index):
    array = []
    for j in range(len(emptyState)):
        char = math.floor(index/(3**(len(emptyState)-1-j))) / 2
        index = index%(3**(len(emptyState)-1-j))
        array.insert(0,char)
    return array
 
        
trainer = Trainer()

trainer.optimize()