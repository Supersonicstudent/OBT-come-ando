input = 1
output_desire = 0

input_weight = 0.5

learning_rate = 0.1

def activation(sum):
    if sum >= 0:
        return 1
    else:
        return 0
    
print ("entrada", input, "desejado", output_desire)

error = 9999999999999999999999999999999999999999999999 


while not error == 0:
    iteration = 1 
    print ("#######interação", iteration)
    print ("peso", input_weight)
    sum = input * input_weight

    output = activation(sum)

    print ("saida", output)

    error = output_desire - output

    print ("erro", error)

    if not error == 0:
        input_weight = input_weight + (learning_rate * input * error)
print("PARABÉNS A REDE APRENDEU")