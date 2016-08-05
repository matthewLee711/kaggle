'''
Purpose of Backpropagation is to find effect of nodes on final output
forward pass - compute the inputs and return the outputs (left to right in circuit). By the end of the
    gates, loss function is performed
backward pass - (right to left) take derivateive identity mapping == 1. SO take
    gradient on z is (opposite) 3. The influence of z on the final value is positive and with
    a force of 3. If increase z by small amout of h, the circuit will react positively by increasing 3h.
    If q were to increase, it would decrease by -4h (slope) (this is all recursive).
    Gate will envuallty learn about influence on final output. Multiply back propagated gradient with input gradient
    to find influence of input. finding the influence every single node on the final output
Local influence of y on q is 1 (from chain rule). The way to put together q and z is to use the chain rule
    and combine them

We havethese two pieces we keep multiplying
'''



'''
add gate: gradient distrbutor
max gate: gradient router
mul gate: gradient switcher
41:40 super Important lecture 4
when you want to do update, you need the gradient.
1. first sample amini batch, then do forward and backward, update

Forward calculates loss
backwards calculates gradient
Need to go backward and forward to update
update then uses
'''


'''

'''

'''
'''


'''
'''
