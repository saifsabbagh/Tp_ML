card_values = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}

def parser():
    while 1:
        data = list(input().split(' '))
        for number in data:
            if len(number) > 0:
                yield(number)   

input_parser = parser()

def get_word():
    global input_parser
    return next(input_parser)

def get_number():
    data = get_word()
    try:
        return int(data)
    except ValueError:
        return float(data)
def game(p1,p2):
    
    while( p1!=[] and p2!=[] and (p1[0]!=p2[0]) ):
        if (card_values[p1[0]]>card_values[p2[0]]):
            p1.append(p2[0])
            p2.pop(p2[0])
            p1.pop(p1[0])
        else:
            p2.append(p1.pop(p1[0]))
            p2.pop(p2[0])
    if (p1==[]):
        print("player 2")
    elif(p2==[]):
        print("player 1")
    else:
        print("draw")

n=get_number()
for i in range(n):
    a=get_number()
    ch1=get_word()
    ch2=get_word()
    game(ch1,ch2)
   

        

        
        
        
        
    
                