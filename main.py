from backend.kinova import BaseApp
import time

class Main(BaseApp):

    def start(self):
        # Go to home
        # Get initial camera frame once at home -- FUNCTION
        #   Camera Calibration
        # Save positions for Pour Cup & Fill Cup in world & camera frame -- FUNCTION
        #   World 0,0,0 at arm base
        # Set state to WAITING

        pass        
        
    def loop(self):        
        # if state == ACTING
        #   do actions until list is complete
        #   Set state to waiting
        # elif state == WAITING
        #   command input method  --  Button / Screen / Etc...
        #   command -> add list of actions
        #       Go to first cup
        #           Trajectory generation
        #           Movement
        #       Pick up cup
        #       Move up
        #           Trajectory generation
        #           Movement
        #       Go to pour location
        #           Trajectory generation
        #           Movement
        #       Pour
        #       Straighten
        #       Move back
        #           Trajectory generation
        #           Movement
        #       Put down bup
        #   Set state to acting
        
        pass


if __name__ == "__main__":
    final_project = Main()

    try:
        while True:
            time.sleep(0.1)  
    except KeyboardInterrupt:
        final_project.shutdown()