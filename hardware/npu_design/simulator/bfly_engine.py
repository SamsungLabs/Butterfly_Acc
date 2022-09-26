import math
import logging
logger = logging.getLogger(__name__)

class bfly_engine:
    def __init__(self, num_bu):
        self.num_bu = num_bu
    
    def run(self, length):
        initial_delay = 0
        compute_cycles = length/(2*self.num_bu) * (length.bit_length()-1)
        logging.debug("Computation takes %d cycles with initial delay %d" % (compute_cycles, initial_delay))
        return compute_cycles
        # return initial_delay, compute_cycles
