import logging
logger = logging.getLogger(__name__)

class Bram:
    def __init__(self, height_per_bank, bitwidth_per_bank, num_bank, bram_name):
        self.num_bank = num_bank
        self.bitwidth_per_bank = bitwidth_per_bank
        self.num_read_access = 0 # Leave for ASIC simulation
        self.num_write_access = 0 # Leave for ASIC simulation
        self.bram_width = self.bitwidth_per_bank * self.num_bank
        self.bram_height = height_per_bank
        self.bram_name = bram_name

    def reset_stat(self):
        self.num_read_access = 0
        self.num_write_access = 0

    def read(self, read_height, read_width, bit_width, num_reuse=1, num_repeat=1):
        #assert (read_width * bit_width) % self.bram_width == 0, "%d, %d, %d" % (read_width, bit_width, self.bram_width)
        read_depth = (read_width * bit_width) / self.bram_width  # If read width * bitwidth larege than bram width, split the data
        assert read_depth * read_height <= self.bram_height
        self.num_read_access += read_depth * read_height * num_repeat
        read_cycle = read_depth * read_height * num_reuse * num_repeat
        logging.debug("Reading %d x %d from Bram %s (bandwidth %d) with reuse %d and repeat %d takes %d cycles" 
                        % (read_height, read_width, self.bram_name, self.bram_width, num_reuse, num_repeat, read_cycle))
        return read_cycle

    def write(self, write_height, write_width, bit_width):
        # assert (write_width * bit_width) % self.bram_width == 0, "%d, %d, %d" % (write_width, bit_width, self.bram_width)
        write_depth = (write_width * bit_width) / self.bram_width  # If read width larege than bram width, split the data
        assert write_depth * write_height <= self.bram_height, "%d, %d, %d" % (write_depth, write_height, self.bram_height)
        self.num_write_access += write_depth * write_height
        write_cycle = write_depth * write_height
        # logging.debug("write_width %d, bit_width %d, bram_width %d" % (write_width, bit_width, self.bram_width))
        # logging.debug("write_depth %d, write_height %d" % (write_depth, write_height))
        logging.debug("Writing %d x %d to Bram %s (bandwidth %d) takes %d cycles" % (write_height, write_width, self.bram_name, self.bram_width, write_cycle))
        return write_cycle
