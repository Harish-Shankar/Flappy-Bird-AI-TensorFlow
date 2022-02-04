from itertools import cycle
import random
import sys
import pygame
from pygame.locals import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

FPS = 60
SCREENWIDTH = 288.0
SCREENHEIGHT = 512.0
PIPEGAPSIZE = 100
PIPESPACESIZE = SCREENHEIGHT * 0.79
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

loadSavedPool = False
saveCurrentPool = True
currentPool = []
fitness = []
totalModels = 50
nextPipeX = -1
nextPipeHoleY = -1
generation = 1
highestFitness = -1
bestWeights = []

def create_model():
    model = Sequential()
    model.add(Dense(3, input_shape=(3,)))
    model.add(Activation('relu'))
    model.add(Dense(7, input_shape=(3,)))
    model.add(Activation('relu'))
    model.add(Dense(1, input_shape=(3,)))
    model.add(Activation('sigmoid'))
    model.compile(loss='mse',optimizer='adam')
    return model

def predict_action(height, dist, pipeHeight, modelNum):
    global currentPool

    height = min(SCREENHEIGHT, height) / SCREENHEIGHT - 0.5
    dist = dist / 450 - 0.5
    pipeHeight = min(SCREENHEIGHT, pipeHeight) / SCREENHEIGHT - 0.5
    neuralInput = np.asarray([height,dist,pipeHeight])
    neuralInput = np.atleast_2d(neuralInput)
    outputProb = currentPool[modelNum].predict(neuralInput, 1)[0]

    if(outputProb[0] <= .5):
        return 1
    return 2

def model_crossover(parent1, parent2):
    global currentPool

    weight1 = currentPool[parent1].get_weights()
    weight2 = currentPool[parent2].get_weights()
    newWeight1 = weight1
    newWeight2 = weight2
    gene = random.randint(0,len(newWeight1)-1)
    newWeight1[gene] = weight2[gene]
    newWeight2[gene] = weight1[gene]

    return np.asarray([newWeight1,newWeight2])

def model_mutate(weights):
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            if( random.uniform(0,1) > .85):
                change = random.uniform(-.5,.5)
                weights[i][j] += change
    return weights

def save_pool():
    for x in range(totalModels):
        currentPool[x].save_weights("SavedModels/model_new" + str(x) + ".keras")
    print("Saved current pool!")


for i in range(totalModels):
    model = create_model()
    currentPool.append(model)
    # reset fitness score
    fitness.append(-100)


if loadSavedPool:
    for i in range(totalModels):
        currentPool[i].load_weights("SavedModels/model_new"+str(i)+".keras")

def main():
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((int(SCREENWIDTH), int(SCREENHEIGHT)))
    pygame.display.set_caption('Flappy Bird')

    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'
    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    while True:
        IMAGES['background'] = pygame.image.load("/assets/sprites/background-day.png").convert()
        IMAGES['player'] = (
            pygame.image.load("/assets/sprites/yellowbird-upflap.png").convert_alpha(),
            pygame.image.load("/assets/sprites/yellowbird-midflap.png").convert_alpha(),
            pygame.image.load("/assets/sprites/yellowbird-downflap.png").convert_alpha(),
        )
        IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load('/assets/sprites/pipe-green.png').convert_alpha(), 180),
            pygame.image.load('/assets/sprites/pipe-green.png').convert_alpha(),
        )

        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )

        movementInfo = showWelcomeAnimation()
        global fitness
        for idx in range(totalModels):
            fitness[idx] = 0
        crashInfo = mainGame(movementInfo)
        showGameOverScreen(crashInfo)

def showWelcomeAnimation():
    return {
                'playery': int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2),
                'basex': 0,
                'playerIndexGen': cycle([0, 1, 2, 1]),
            }

def mainGame(movementInfo):
    global fitness
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playersXList = []
    playersYList = []
    for idx in range(totalModels):
        playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']
        playersXList.append(playerx)
        playersYList.append(playery)
    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    global nextPipeX
    global nextPipeHoleY

    nextPipeX = lowerPipes[0]['x']
    nextPipeHoleY = (lowerPipes[0]['y'] + (upperPipes[0]['y'] + IMAGES['pipe'][0].get_height()))/2

    pipeVelX = -4
    playersVelY    =  [] 
    playerMaxVelY =  10
    playerMinVelY =  -8
    playersAccY    =  []
    playerFlapAcc =  -9
    playersFlapped = []
    playersState = []

    for idx in range(totalModels):
        playersVelY.append(-9)
        playersAccY.append(1)
        playersFlapped.append(False)
        playersState.append(True)

    alive_players = totalModels

    while True:
        for idxPlayer in range(totalModels):
            if playersYList[idxPlayer] < 0 and playersState[idxPlayer] == True:
                alive_players -= 1
                playersState[idxPlayer] = False
        if alive_players == 0:
            return {
                'y': 0,
                'groundCrash': True,
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': 0,
            }
        for idxPlayer in range(totalModels):
            if playersState[idxPlayer]:
                fitness[idxPlayer] += 1
        nextPipeX += pipeVelX
        for idxPlayer in range(totalModels):
            if playersState[idxPlayer]:
                if predict_action(playersYList[idxPlayer], nextPipeX, nextPipeHoleY, idxPlayer) == 1:
                    if playersYList[idxPlayer] > -2 * IMAGES['player'][0].get_height():
                        playersVelY[idxPlayer] = playerFlapAcc
                        playersFlapped[idxPlayer] = True
                        SOUNDS['wing'].play()
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

        crashTest = checkCrash({'x': playersXList, 'y': playersYList, 'index': playerIndex},
                               upperPipes, lowerPipes)

        for idx in range(totalModels):
            if playersState[idx] == True and crashTest[idx] == True:
                alive_players -= 1
                playersState[idx] = False
        if alive_players == 0:
            return {
                'y': playery,
                'groundCrash': crashTest[1],
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': 0,
            }

        goneThroughAPipe = False
        for idx in range(totalModels):
            if playersState[idx] == True:
                pipeIndex = 0
                playerMidPos = playersXList[idx]
                for pipe in upperPipes:
                    pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width()
                    if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                        nextPipeX = lowerPipes[pipeIndex+1]['x']
                        nextPipeHoleY = (lowerPipes[pipeIndex+1]['y'] + (upperPipes[pipeIndex+1]['y'] + IMAGES['pipe'][pipeIndex+1].get_height())) / 2
                        goneThroughAPipe  = True
                        score += 1
                        fitness[idx] += 25
                        SOUNDS['point'].play()
                    pipeIndex += 1

        if(goneThroughAPipe):
            score += 1

        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        for idx in range(totalModels):
            if playersState[idx] == True:
                if playersVelY[idx] < playerMaxVelY and not playersFlapped[idx]:
                    playersVelY[idx] += playersAccY[idx]
                if playersFlapped[idx]:
                    playersFlapped[idx] = False
                playerHeight = IMAGES['player'][playerIndex].get_height()
                playersYList[idx] += min(playersVelY[idx], PIPESPACESIZE - playersYList[idx] - playerHeight)

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, PIPESPACESIZE))
        showScore(score)
        for idx in range(totalModels):
            if playersState[idx] == True:
                SCREEN.blit(IMAGES['player'][playerIndex], (playersXList[idx], playersYList[idx]))

        pygame.display.update()
        FPSCLOCK.tick(FPS)
    
def getRandomPipe():
    gapY = random.randrange(0, int(PIPESPACESIZE * 0.6 - PIPEGAPSIZE))
    gapY += int(PIPESPACESIZE * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},
    ]

def checkCrash(players, upperPipes, lowerPipes):
    statuses = []
    for idx in range(totalModels):
        statuses.append(False)

    for idx in range(totalModels):
        statuses[idx] = False
        pi = players['index']
        players['w'] = IMAGES['player'][0].get_width()
        players['h'] = IMAGES['player'][0].get_height()
        if players['y'][idx] + players['h'] >= PIPESPACESIZE - 1:
            statuses[idx] = True
        playerRect = pygame.Rect(players['x'][idx], players['y'][idx],
                      players['w'], players['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                statuses[idx] = True
    return statuses

def showScore(score):
    global generation
    scoreDigits = [int(x) for x in list(str(score))]
    generation_digits = [int(x) for x in list(str(generation))]
    totalWidth1 = 0
    totalWidth2 = 0

    for digit in scoreDigits:
        totalWidth1 += IMAGES['numbers'][digit].get_width()

    for digit in generation_digits:
        totalWidth2 += IMAGES['numbers'][digit].get_width()

    Xoffset1 = (SCREENWIDTH - totalWidth1) / 2
    Xoffset2 = (SCREENWIDTH - totalWidth2) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset1, SCREENHEIGHT * 0.1))
        Xoffset1 += IMAGES['numbers'][digit].get_width()

    for digit in generation_digits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset2, SCREENHEIGHT * 0.2))
        Xoffset2 += IMAGES['numbers'][digit].get_width()

def getHitmask(image):
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    rect = rect1.clip(rect2)
    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def showGameOverScreen(crashInfo):
    global currentPool
    global fitness
    global generation
    newWeights = []
    totalFitness = 0
    global highestFitness
    global bestWeights
    updated = False

    for select in range(totalModels):
        totalFitness += fitness[select]
        if fitness[select] >= highest_fitness:
            updated = True
            highest_fitness = fitness[select]
            best_weights = currentPool[select].get_weights()

    parent1 = random.randint(0,totalModels-1)
    parent2 = random.randint(0,totalModels-1)

    for i in range(totalModels):
        if fitness[i] >= fitness[parent1]:
            parent1 = i

    for j in range(totalModels):
        if j != parent1:
            if fitness[j] >= fitness[parent2]:
                parent2 = j


    for select in range(totalModels // 2):
        cross_over_weights = model_crossover(parent1,parent2)
        if updated == False:
            cross_over_weights[1] = best_weights
        mutated1 = model_mutate(cross_over_weights[0])
        mutated2 = model_mutate(cross_over_weights[0])

        newWeights.append(mutated1)
        newWeights.append(mutated2)

    for select in range(len(newWeights)):
        fitness[select] = -100
        currentPool[select].set_weights(newWeights[select])
    if saveCurrentPool == 1:
        save_pool()

    generation += 1
    return

main()