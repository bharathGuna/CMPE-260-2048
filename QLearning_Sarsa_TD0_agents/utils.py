
import cv2
import numpy as np

def randArgMax(a):
    '''Returns the argmax of the array. Ties are broken radnomly.'''
    return np.argmax(np.random.random(np.shape(a))*(a==np.max(a)))
    
def makeImage(score, state, board_size=4, graphic_size=750, top_margin=40,
              seperator_width=12):
    '''Construct the image for a game state
    input:
        score: Score of the game
        state: Board state
        board_size: Number of tiles in one side of board
        graphic_size: Size of graphic
        top_margin: Size of top margin
        seperator_width: Seperation between tiles in graphic
    output: Image for a game state'''
    img = np.full((graphic_size + top_margin, graphic_size, 3), 255,
                     np.uint8)
    # Define colors
    background_color = (146, 135, 125)
    color = {0:(158, 148, 138), 1:(238, 228, 218), 2:(237, 224, 200),
             3:(242, 177, 121), 4:(245, 149, 99), 5:(246, 124, 95), 
             6:(246, 94, 59), 7:(237, 207, 114), 8:(237, 204, 97), 
             9:(237, 200, 80), 10:(237, 197, 63), 11:(237, 197, 63), 
             12:(62, 237, 193), 13:(62, 237, 193), 14:(62,64,237), 
             15:(140,62,237)}
    #Set font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Define spacing of tiles
    spacing = int((graphic_size-seperator_width)/board_size)
    # Write score at top of screen
    text = 'The score is ' + str(score)
    textsize = cv2.getTextSize(text, font, 0.5, 1)[0]
    cv2.putText(img,text,(int((graphic_size-textsize[0])/2),
                          int((3*top_margin/4+textsize[1])/2)),
                font,0.5,(0,0,0),1,cv2.LINE_AA)
    # Draw squares
    for i in range(4):
        for k in range(4):
            cv2.rectangle(img,
                          (int(seperator_width/2)+k*spacing,
                           int(top_margin+seperator_width/2)+i*spacing),
                          (int(seperator_width/2)+(k+1)*spacing,
                           int(top_margin+seperator_width/2)+(i+1)*spacing),
                          color[state[i][k]], -1)
            if state[i][k] == 0:
                text = ''
            else:
                text = str(2**state[i][k])
            textsize = cv2.getTextSize(text, font, 0.5, 2)[0]
            cv2.putText(img,text,
                        (int(seperator_width/2+k*spacing+(spacing-textsize[0])/2),
                         int(top_margin+seperator_width/2+i*spacing+(spacing+textsize[1])/2)),
                        font,0.5,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(img,text,(int(seperator_width/2+k*spacing+(spacing-textsize[0])/2),
                                  int(top_margin+seperator_width/2+i*spacing+(spacing+textsize[1])/2)),
                        font,0.5,(255,255,255),1,cv2.LINE_AA)
    # Draw outline grid
    for i in range(5):
        cv2.line(img, 
                (int(seperator_width/2)+i*spacing,int(top_margin+seperator_width/2)),
                (int(seperator_width/2)+i*spacing,int(graphic_size+top_margin-seperator_width/2)), 
                 background_color, seperator_width)
    for i in range(5):
        cv2.line(img,
                 (int(seperator_width/2),int(top_margin+seperator_width/2)+i*spacing),
                 (int(graphic_size-seperator_width/2),int(top_margin+seperator_width/2)+i*spacing),
                 background_color,seperator_width)
    return img