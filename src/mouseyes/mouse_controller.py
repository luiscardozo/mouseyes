'''
This is a sample class that you can use to control the mouse pointer.
It uses the pyautogui library. You can set the precision for mouse movement
(how much the mouse moves) and the speed (how fast it moves) by changing 
precision_dict and speed_dict.
Calling the move function with the x and y output of the gaze estimation model
will move the pointer.
This class is provided to help get you started; you can choose whether you want to use it or create your own from scratch.
'''
import pyautogui
import time

class MouseController:
    def __init__(self, precision, speed, logger):
        precision_dict={'high':100, 'low':1000, 'medium':500}
        speed_dict={'fast':1, 'slow':10, 'medium':5}
        self.screenWidth, self.screenHeight = pyautogui.size()
        #print(f"wxh: {self.screenWidth}x{self.screenHeight}")

        self.precision=precision_dict[precision]
        self.speed=speed_dict[speed]
        self.logger = logger

    def safe_rel_move(self, relMove, curPos, limit):
        relMove = int(relMove*self.precision)
        moveToDir = curPos + relMove
        #print(f"relMove: {relMove},  moveToDir: {moveToDir}, curPos: {curPos}, limit: {limit}")
        if moveToDir < 0:
            relMove = -curPos +3
        elif moveToDir >= limit:
            relMove = limit - curPos -3
        #print(f"NewRelMove: {relMove}, moveTo: {relMove+curPos}")
        return relMove

    def move(self, x, y):
        curPosX, curPosY = pyautogui.position()

        moveX = self.safe_rel_move(x, curPosX, self.screenWidth)
        moveY = self.safe_rel_move(-1*y, curPosY, self.screenHeight)

        #print("moving mouse to {}, {} (curPos:{}x{}. Sum:{}x{})".format(moveX, moveY, curPosX, curPosY, curPosX+moveX, curPosY+moveY))

        self.logger.debug("moving mouse to {}, {}".format(moveX, moveY))
        pyautogui.moveRel(moveX, moveY, duration=self.speed)

    def check_boundaries(self, axis, limit):
        if(axis < 0):
            axis = 0
        else:
            if axis > limit:
                axis = limit
        return axis

    def safe_move(self, x, y):
        x = self.check_boundaries(x, self.screenWidth)
        y = self.check_boundaries(y, self.screenHeight)
        return x, y

    def moveTo(self, x, y, duration=1, tween=pyautogui.easeInOutQuad):
        x, y = self.safe_move(x, y)
        pyautogui.moveTo(x, y, duration, tween)

    def dragTo(self, x, y, button='left', duration=2):
        x, y = self.safe_move(x, y)
        pyautogui.dragTo(x, y, duration, button=button)

    def drag(self, x, y, button='left', duration=2):
        x, y = self.safe_move(x, y)
        pyautogui.drag(x, y, duration, button)

    def click(self, doubleClick=False):
        if doubleClick:
            pyautogui.doubleClick()
        else:
            pyautogui.click()

if __name__ == "__main__":
    mc = MouseController('medium', 'fast')

    screenWidth, screenHeight = pyautogui.size()
    print(f"screen: {screenWidth}x{screenHeight}")

    currentMouseX, currentMouseY = pyautogui.position()
    print(f"current pos: {currentMouseX}, {currentMouseY}")
    #mc.moveTo(100,500)
    #time.sleep(2)
    mc.dragTo(100, 100)