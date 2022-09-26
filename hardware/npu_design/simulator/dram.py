import math
import logging
logger = logging.getLogger(__name__)

class Dram:
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
        self.num_read_access = 0 # Leave for ASIC simulation
        self.num_write_access = 0 # Leave for ASIC simulation

    def reset_stat(self):
        self.num_read_access = 0
        self.num_write_access = 0

    def read(self, read_height, read_width, bit_width): # Continuos read
        assert self.bandwidth % bit_width == 0
        pack_read_factor = self.bandwidth // bit_width
        self.num_read_access += int(math.ceil(read_width * read_height / pack_read_factor))
        read_cycle = int(math.ceil(read_width * read_height / pack_read_factor))
        logging.debug("Reading %d x %d from Dram (bandwidth %d) takes %d cycles"%(read_height, read_width, self.bandwidth, read_cycle))
        return read_cycle

    def write(self, write_height, write_width, bit_width): # Continuos read
        assert self.bandwidth % bit_width == 0
        pack_write_factor = self.bandwidth // bit_width
        self.num_write_access += int(math.ceil(write_width * write_height / pack_write_factor))
        write_cycle = int(math.ceil(write_width * write_height / pack_write_factor))
        logging.debug("Writing %d x %d from Dram (bandwidth %d) to Dram takes %d cycles"%(write_height, write_width, self.bandwidth, write_cycle))
        return write_cycle
