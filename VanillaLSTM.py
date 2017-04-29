import numpy as np

class RecurrentNeuralNetwork:
    
    def __init__ (self, xs, ys, rl, eo, lr):
        #initial input 
        self.x = np.zeros(xs)
        #input size
        self.xs = xs
        #expected output 
        self.y = np.zeros(ys)
        #output size
        self.ys = ys
        #weight matrix for interpreting results from LSTM cell
        self.w = np.random.random((ys, ys))
        #matrix used in RMSprop
        self.G = np.zeros_like(self.w)
        #length of the recurrent network - number of recurrences
        self.rl = rl
        #learning rate
        self.lr = lr
        #array for storing inputs
        self.ia = np.zeros((rl+1,xs))
        #array for storing cell states
        self.ca = np.zeros((rl+1,ys))
        #array for storing outputs
        self.oa = np.zeros((rl+1,ys))
        #array for storing hidden states
        self.ha = np.zeros((rl+1,ys))
        self.af = np.zeros((rl+1,ys))
        self.ai = np.zeros((rl+1,ys))
        self.ac = np.zeros((rl+1,ys))
        self.ao = np.zeros((rl+1,ys))
        #array of expected output values
        self.eo = np.vstack((np.zeros(eo.shape[0]), eo.T))
        #declare LSTM cell - add ability to declare multiple cells in future
        self.LSTM = LSTM(xs, ys, rl, lr)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))    
            
    def forwardProp(self):
        #loop through recurrences - start at 1 so the 0th entry of all arrays will be an array of 0's
        for i in range(1, self.rl+1):
            #set input for LSTM cell, combination of input (previous output) and previous hidden state
            self.LSTM.x = np.hstack((self.ha[i-1], self.x))
            #run forward prop on the LSTM cell, retrieve cell state and hidden state
            cs, hs, f, inp, c, o = self.LSTM.forwardProp()
            #store input
            maxI = np.argmax(self.x)
            self.x = np.zeros_like(self.x)
            self.x[maxI] = 1
            self.ia[i] = self.x #Use np.argmax?
            #store cell state
            self.ca[i] = cs
            #store hidden state
            self.ha[i] = hs
            self.af[i] = f
            self.ai[i] = inp
            self.ac[i] = c
            self.ao[i] = o
            #calculate output by multiplying hidden state with weight matrix
            self.oa[i] = self.sigmoid(np.dot(self.w, hs))
            self.x = self.eo[i-1]
        #return all outputs    
        return self.oa
    
    def backProp(self):
        totalError = 0
        dfcs = np.zeros(self.ys)
        dfhs = np.zeros(self.ys)
        #initialize matrices for gradient updates
        tfu = np.zeros((self.ys, self.xs+self.ys))
        tiu = np.zeros((self.ys, self.xs+self.ys))
        tcu = np.zeros((self.ys, self.xs+self.ys))
        tou = np.zeros((self.ys, self.xs+self.ys))
        tu = np.zeros((self.ys,self.ys))
        #loop backwards through recurrences
        for i in range(self.rl, -1, -1):
            #error = calculatedOutput - expectedOutput
            error = self.oa[i] - self.eo[i]
            #calculate update for weight matrix
            tu += np.dot(np.atleast_2d(error * self.dsigmoid(self.oa[i])), np.atleast_2d(self.ha[i]).T)
            #propagate error back to exit of LSTM cell
            error = np.dot(error, self.w)
            #set input values of LSTM cell for recurrence i
            self.LSTM.x = np.hstack((self.ha[i-1], self.ia[i]))
            #set cell state of LSTM cell for recurrence i
            self.LSTM.cs = self.ca[i]
            #call the LSTM cell's backprop, retreive gradient updates
            fu, iu, cu, ou, dfcs, dfhs = self.LSTM.backProp(error, self.ca[i-1], self.af[i], self.ai[i], self.ac[i], self.ao[i], dfcs, dfhs)
            #calculate total error (not necesarry, used to measure training progress)
            totalError += np.sum(error)
            #accumulate all gradient updates
            tfu += fu
            tiu += iu
            tcu += cu
            tou += ou
        #update LSTM matrices with average of accumulated gradient updates    
        self.LSTM.update(tfu/self.rl, tiu/self.rl, tcu/self.rl, tou/self.rl) 
        #update weight matrix with average of accumulated gradient updates  
        self.update(tu/self.rl)
        #return total error of this iteration
        return totalError
    
    def update(self, u):
        #vanilla implementation of RMSprop
        self.G = 0.9 * self.G + 0.1 * u**2  
        self.w -= self.lr/np.sqrt(self.G + 1e-8) * u
        return
    
    def sample(self):
         #loop through recurrences - start at 1 so the 0th entry of all arrays will be an array of 0's
        for i in range(1, self.rl+1):
            #set input for LSTM cell, combination of input (previous output) and previous hidden state
            self.LSTM.x = np.hstack((self.ha[i-1], self.x))
            #run forward prop on the LSTM cell, retrieve cell state and hidden state
            cs, hs, f, inp, c, o = self.LSTM.forwardProp()
            #store input
            maxI = np.argmax(self.x)
            self.x = np.zeros_like(self.x)
            self.x[maxI] = 1
            self.ia[i] = self.x #Use np.argmax?
            #store cell state
            self.ca[i] = cs
            #store hidden state
            self.ha[i] = hs
            self.af[i] = f
            self.ai[i] = inp
            self.ac[i] = c
            self.ao[i] = o
            #calculate output by multiplying hidden state with weight matrix
            self.oa[i] = self.sigmoid(np.dot(self.w, hs))
            maxI = np.argmax(self.oa[i])
            newX = np.zeros_like(self.x)
            newX[maxI] = 1
            self.x = newX
        #return all outputs    
        return self.oa
        
class LSTM:
    
    def __init__ (self, xs, ys, rl, lr):
        self.x = np.zeros(xs+ys)
        self.xs = xs + ys
        self.y = np.zeros(ys)
        self.ys = ys
        self.cs = np.zeros(ys)
        self.rl = rl
        self.lr = lr
        self.f = np.random.random((ys, xs+ys))
        self.i = np.random.random((ys, xs+ys))
        self.c = np.random.random((ys, xs+ys))
        self.o = np.random.random((ys, xs+ys))
        self.Gf = np.zeros_like(self.f)
        self.Gi = np.zeros_like(self.i)
        self.Gc = np.zeros_like(self.c)
        self.Go = np.zeros_like(self.o)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def tangent(self, x):
        return np.tanh(x)
    
    def dtangent(self, x):
        return 1 - np.tanh(x)**2
    
    def forwardProp(self):
        f = self.sigmoid(np.dot(self.f, self.x))
        self.cs *= f
        i = self.sigmoid(np.dot(self.i, self.x))
        c = self.tangent(np.dot(self.c, self.x))
        self.cs += i * c
        o = self.sigmoid(np.dot(self.o, self.x))
        self.y = o * self.tangent(self.cs)
        return self.cs, self.y, f, i, c, o
   
    def backProp(self, e, pcs, f, i, c, o, dfcs, dfhs):
        e = np.clip(e + dfhs, -6, 6)
        do = self.tangent(self.cs) * e
        ou = np.dot(np.atleast_2d(do * self.dtangent(o)).T, np.atleast_2d(self.x))
        dcs = np.clip(e * o * self.dtangent(self.cs) + dfcs, -6, 6)
        dc = dcs * i
        cu = np.dot(np.atleast_2d(dc * self.dtangent(c)).T, np.atleast_2d(self.x))
        di = dcs * c
        iu = np.dot(np.atleast_2d(di * self.dsigmoid(i)).T, np.atleast_2d(self.x))
        df = dcs * pcs
        fu = np.dot(np.atleast_2d(df * self.dsigmoid(f)).T, np.atleast_2d(self.x))
        dpcs = dcs * f
        dphs = np.dot(dc, self.c)[:self.ys] + np.dot(do, self.o)[:self.ys] + np.dot(di, self.i)[:self.ys] + np.dot(df, self.f)[:self.ys] 
        return fu, iu, cu, ou, dpcs, dphs
            
    def update(self, fu, iu, cu, ou):
        self.Gf = 0.9 * self.Gf + 0.1 * fu**2 
        self.Gi = 0.9 * self.Gi + 0.1 * iu**2   
        self.Gc = 0.9 * self.Gc + 0.1 * cu**2   
        self.Go = 0.9 * self.Go + 0.1 * ou**2   
        self.f -= self.lr/np.sqrt(self.Gf + 1e-8) * fu
        self.i -= self.lr/np.sqrt(self.Gi + 1e-8) * iu
        self.c -= self.lr/np.sqrt(self.Gc + 1e-8) * cu
        self.o -= self.lr/np.sqrt(self.Go + 1e-8) * ou
        return
        
def LoadText():
    with open("C:\\Users\\Kevin\\Documents\\shakespeare.txt", "r") as text_file:
        data = text_file.read()
    text = list(data)
    outputSize = len(text)
    data = list(set(text))
    uniqueWords, dataSize = len(data), len(data) 
    returnData = np.zeros((uniqueWords, dataSize))
    for i in range(0, dataSize):
        returnData[i][i] = 1
    returnData = np.append(returnData, np.atleast_2d(data), axis=0)
    output = np.zeros((uniqueWords, outputSize))
    for i in range(0, outputSize):
        index = np.where(np.asarray(data) == text[i])
        output[:,i] = returnData[0:-1,index[0]].astype(float).ravel()  
    return returnData, uniqueWords, output, outputSize, data

def ExportText(output, data):
    finalOutput = np.zeros_like(output)
    prob = np.zeros_like(output[0])
    outputText = ""
    print(len(data))
    print(output.shape[0])
    for i in range(0, output.shape[0]):
        for j in range(0, output.shape[1]):
            prob[j] = output[i][j] / np.sum(output[i])
        outputText += np.random.choice(data, p=prob)    
    with open("C:\\Users\\Kevin\\Documents\\output.txt", "w") as text_file:
        text_file.write(outputText)
    return

#Begin program    
print("Beginning")
iterations = 5000
learningRate = 0.001
returnData, numCategories, expectedOutput, outputSize, data = LoadText()
print("Done Reading")
RNN = RecurrentNeuralNetwork(numCategories, numCategories, outputSize, expectedOutput, learningRate)

for i in range(1, iterations):
    RNN.forwardProp()
    error = RNN.backProp()
    print("Error on iteration ", i, ": ", error)
    if error > -100 and error < 100 or i % 100 == 0:
        seed = np.zeros_like(RNN.x)
        maxI = np.argmax(np.random.random(RNN.x.shape))
        seed[maxI] = 1
        RNN.x = seed  
        output = RNN.sample()
        print(output)    
        ExportText(output, data)
        print("Done Writing")
print("Complete")
