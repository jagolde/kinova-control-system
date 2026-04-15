from backend.kinova import Kinova
import sys, time
import numpy as np
import queue, threading

class Main:
    
    HOME_POSITION = [
        np.radians(100),
        np.radians(330),
        np.radians(125),
        np.radians(140),
        np.radians(260),
        np.radians(0),
    ]
    
    N = [
        np.radians(45),
        np.radians(350),
        np.radians(85),
        np.radians(80),
        np.radians(350),
        np.radians(90),
    ]
    
    
    def __init__(self, loop_rate = 20) -> None:
        self.kinova_robot = Kinova()
        self.LOOP_RATE = 1 / float(loop_rate)
        
        self.action_queue = queue.Queue()

        self.is_running = True
        
        self.start()
        
        self.background_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.background_thread.start()
        print("loop Loop Started")
            
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
    
    # DO NOT TOUCH
    def _start_loop(self):
        try:
            while self.is_running:
                if not self.action_queue.empty():
                    func, args = self.action_queue.get()
                    print(f'Executing: {func.__name__}')
                    func(*args)
                self.loop()
                time.sleep(self.LOOP_RATE)
        except Exception as e:
            print(f'ERROR Background loop crashed: {e}')
            
            
    # DO NOT TOUCH
    def shutdown(self):
        print("Shutting down gracefully")
        self.is_running = False
        self.kinova_robot.set_torque(True)
        self.kinova_robot.stop()
        sys.exit(0)
            

if __name__ == "__main__":
    final_project = Main()
    # KEEP THIS LOOP
    try:
        while True:
            pass
    except KeyboardInterrupt:
        final_project.shutdown()