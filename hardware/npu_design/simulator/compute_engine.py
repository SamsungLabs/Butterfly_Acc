import math
import logging
logger = logging.getLogger(__name__)

class CE:
    def __init__(self, num_mults_per_pe, num_pes, ce_name, mult_lat=2, add_lat=1, reg_initial_delay=2, reg_mult_pipeline_delay=3, reg_add_pipeline_delay=1):
        self.mult_lat = mult_lat
        self.add_lat = add_lat
        self.acc_lat = add_lat
        self.bias_add_lat = add_lat
        self.num_mults_per_pe = num_mults_per_pe
        self.num_pes = num_pes
        self.ce_name = ce_name
        self.reg_initial_delay = reg_initial_delay
        self.reg_mult_pipeline_delay = reg_mult_pipeline_delay
        self.reg_add_pipeline_delay = reg_add_pipeline_delay
    
    def run(self, num_row, vec_len, num_column):
        num_acc = (vec_len / self.num_mults_per_pe)
        initial_delay = (self.mult_lat + self.reg_mult_pipeline_delay + (self.add_lat + self.reg_add_pipeline_delay) * math.log(self.num_mults_per_pe, 2)
                           + self.acc_lat + self.reg_initial_delay + self.bias_add_lat)# + num_acc-1)
        compute_cycles = (num_row * num_column * vec_len) / (self.num_mults_per_pe * self.num_pes)
        logging.debug("Computation of (%d, %d) x (%d, %d) in %s takes %d cycles with initial delay %d" % (num_row, num_column, num_column, vec_len, self.ce_name, compute_cycles, initial_delay))
        return initial_delay, compute_cycles
