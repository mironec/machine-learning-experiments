mutable struct Perceptron
    weightsIn
    valueOut
end

struct Pattern
    inputValues
    outputValues
end

function sigmoid(x::AbstractFloat)::AbstractFloat
    return 1/(1+exp(-x))
end

function stepF(x::AbstractFloat)::AbstractFloat
    if x >= 0
        return 1
    end
    return -1
end

function epoch(patterns, neurons)::AbstractFloat
    numLayers = length(neurons)
    numPatterns = length(patterns)
    errorAcc = 0.0
    for p = 1:numPatterns
        neurons[1] = Array{Perceptron}(length(patterns[p].inputValues))
        for i = 1:length(patterns[p].inputValues)
            neurons[1][i] = Perceptron([], patterns[p].inputValues[i])
        end
        #feedforward
        for l = 2:numLayers
            for i = 1:length(neurons[l])
                neurons[l][i].valueOut = 0
                for j = 1:length(neurons[l-1])
                    neurons[l][i].valueOut += neurons[l-1][j].valueOut * neurons[l][i].weightsIn[j]
                end
                neurons[l][i].valueOut = stepF(neurons[l][i].valueOut)
            end
        end
        #evaluate
        errorAcc2 = 0.0
        for i = 1:length(neurons[numLayers])
            errorAcc2 += (patterns[p].outputValues[i] - neurons[numLayers][i].valueOut) ^ 2.0
            print("Expected $(patterns[p].outputValues[i]), got $(neurons[numLayers][i].valueOut).\n")
        end
        errorAcc2 = sqrt(errorAcc2 / length(neurons[numLayers]))
        errorAcc += errorAcc2
    end
    errorAcc /= numPatterns
    return errorAcc
end

function createNeuron(numPreviousLayers)
    return Perceptron([rand() for i=1:numPreviousLayers], 0)
end

function train(patterns, layerNums = [2,1], numEpochs = 1)
    currentEpoch = 1
    neurons = Array{Array{Perceptron}}(length(layerNums))
    for i = 2:length(layerNums)
        neurons[i] = [createNeuron(layerNums[i-1]) for j=1:layerNums[i]]
    end
    while(currentEpoch <= numEpochs)
        error = epoch(patterns, neurons)
        print("Epoch $currentEpoch: error $error\n")
        currentEpoch += 1
    end
end

patterns = [Pattern([0,0], [0]),
            Pattern([0,1], [1]),
            Pattern([1,0], [1]),
            Pattern([1,1], [1])]
train(patterns)
