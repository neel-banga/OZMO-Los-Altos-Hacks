import NLP
import yolo
import CNN
from audio import say
from audio import listen
import text

say('POWERING UP!')
'''
while True:
    command = listen()

    if 'somi' in command:
        break'''

command = listen()

print(command)

label = NLP.classify_input(command)

print(label)

if label == 0:
    say('What would you like to send')
    text1 = listen()
    text.text(text1)

elif label == 1:
    yolo.yolo()

elif label == 2:
    pass
    #CNN.CNN()