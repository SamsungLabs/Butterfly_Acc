import enum
from bram import Bram
from dram import Dram
from bfly_engine import bfly_engine
import logging
import math

logger = logging.getLogger(__name__)

def ceil_power2(x):
    shift_bits = (x-1).bit_length() - 1
    return 2 << shift_bits

class Butterfly_Accelerator:
    def __init__(self, head_dim, hidden_dim, num_len, ffn_inner_dim, parallesm_bu=4, parallesm_be=128, bit_width=16, 
                    indata_dram_bw=2048, coef_dram_bw=256, outdata_dram_bw=2048):
        self.head_dim = head_dim
        self.hidden_dim = ceil_power2(hidden_dim)
        self.ffn_inner_dim = ceil_power2(ffn_inner_dim)
        self.num_len = num_len
        self.parallesm_bu = parallesm_bu # parallelism of butterfly unite
        self.parallesm_be = parallesm_be # parallelism row of butterfly engine
        self.bit_width = bit_width
        self.indata_dram_bw = indata_dram_bw
        self.coef_dram_bw = coef_dram_bw
        self.outdata_dram_bw = outdata_dram_bw
        self.run_cycles = 0
        self.max_length = max(ceil_power2(self.ffn_inner_dim), ceil_power2(self.num_len))
        self.fft_pipeline_stage = 2
        self.bfly_pipeline_stage = 3
        ############################# Define Dram #############################
        self.indata_dram = Dram(self.indata_dram_bw)
        self.coef_dram = Dram(self.coef_dram_bw)
        self.outdata_dram = Dram(self.outdata_dram_bw)
        ############################# Define Bram #############################
        # Each Butterfly engine has two bram banks for Pingpong or Complex/Real
        # Each bram bank has (2*parallesm_bu) bram. Each bram has width "bit_width" and depth "max_length/(2*parallesm_bu)"
        # height, width_per_bank, number of bank
        self.data_bram_a = [Bram(self.max_length/(2*self.parallesm_bu), self.bit_width, 2*self.parallesm_bu, "data_bram_a %d"%i) for i in range(self.parallesm_be)]
        self.data_bram_b = [Bram(self.max_length/(2*self.parallesm_bu), self.bit_width, 2*self.parallesm_bu, "data_bram_b %d"%i) for i in range(self.parallesm_be)]

        # Coef BRAM with size hidden_dim * p_h * head_dim * bit_with
        self.coef_bram = Bram((2*self.max_length)/(4*self.parallesm_bu)*((self.max_length).bit_length()-1), 2*self.bit_width, 4*self.parallesm_bu, "coef_bram") # complex + real
        
        ####################### Define Compute Engine #########################
        self.bfly_engines = [bfly_engine(self.parallesm_bu) for i in range(self.parallesm_be)]
        logging.info("Numb of butterfly unit per butterply is %d, Numb of butterfly engine is %d"%(self.parallesm_bu, self.parallesm_be))

    def reset_stat(self):
        self.run_cycles = 0

    def run_fft(self, is_last=False, complex_input=False, complex_output=False):
        current_cycle = self.run_cycles
        logging.info("Running Fourier layer")
        num_run = int(math.ceil(float(self.num_len) / self.parallesm_be))
        ############################# First Level Pipelining #############################
        # Get data from dram to bram
        if (complex_input): dram_data_read_cycles = self.indata_dram.read(self.hidden_dim, self.parallesm_be, 2*self.bit_width) # Read from dram, complex + real
        else: dram_data_read_cycles = self.indata_dram.read(self.hidden_dim, self.parallesm_be, self.bit_width) # Read from dram
        for i in range(self.parallesm_be):
            self.data_bram_a[i].write(self.hidden_dim, 1, self.bit_width)
            self.data_bram_b[i].write(self.hidden_dim, 1, self.bit_width)
        bram_data_write_cycles = self.hidden_dim #  Due to Serial to Parallel module, input data comes in one by one
        input_data_cycles = max(dram_data_read_cycles, bram_data_write_cycles) 

        # Get coef from dram to bram
        dram_coef_read_cycles = self.coef_dram.read((self.hidden_dim).bit_length()-1, self.hidden_dim, 2*self.bit_width) # symmetric, Log(N) * N parameters, complex + real
        bram_coef_write_cycles = self.coef_bram.write((self.hidden_dim).bit_length()-1, self.hidden_dim, 2*self.bit_width) # symmetric, Log(N) * N parameters, complex + real
        weight_data_cycles = max(dram_coef_read_cycles, bram_coef_write_cycles)
        print ("input_data transfer cycles:", input_data_cycles)
        print ("weight_data transfer cycles:", weight_data_cycles)
        # Start to compute FFT
        bram_data_read_cycles = []
        for i in range(self.parallesm_be):
            bram_a_data_read_cycles = 0
            bram_b_data_read_cycles = 0
            for j in range((self.hidden_dim).bit_length()-1):
                bram_a_data_read_cycles += self.data_bram_a[i].read(self.hidden_dim, 1, self.bit_width)
                bram_b_data_read_cycles += self.data_bram_b[i].read(self.hidden_dim, 1, self.bit_width)
            bram_data_read_cycles.append(bram_a_data_read_cycles)
            bram_data_read_cycles.append(bram_b_data_read_cycles)
        bram_data_read_cycles = max(bram_data_read_cycles)

        fft_compute_time = self.bfly_engines[0].run(self.hidden_dim) # Run in parallel, get the first one
        fft_time = max(fft_compute_time, bram_data_read_cycles)

        print ("fft compute cycles:", fft_time)
        fst_pipeline_cost = fft_time + max(input_data_cycles, weight_data_cycles)
        logging.info("First-level Pipeline: Loading data/coef from Dram and fft computetakes %d cycles", fst_pipeline_cost)

        fst_pipeline_costs = [fst_pipeline_cost for i in range(num_run)]
        # Padding
        for i in range(self.fft_pipeline_stage-1): fst_pipeline_costs.append(0)
        self.run_cycles += fst_pipeline_costs[0]  # Pipeline, obtain max cycles as the real cycle 

        # Output data from bram to dram
        if (complex_output): dram_data_write_cycles = self.outdata_dram.write(self.hidden_dim, self.parallesm_be, 2*self.bit_width) # Write dram, real + complex
        else: dram_data_write_cycles = self.outdata_dram.write(self.hidden_dim, self.parallesm_be, self.bit_width) # Write dram, real + complex
        for i in range(self.parallesm_be):
            self.data_bram_a[i].read(self.hidden_dim, 1, self.bit_width)
            self.data_bram_b[i].read(self.hidden_dim, 1, self.bit_width)
        bram_data_read_cycles =  self.hidden_dim # Due to Parallel to Serial module, output one by one
        output_data_cycles = max(dram_data_write_cycles, bram_data_read_cycles)

        snd_pipeline_costs = [output_data_cycles for i in range(num_run)]
        
        ############################# Calcumulate stage by stage in total #############################
        # if not the last layer, the output can be also overlap with the input of next layer, the time spend on last outputing can be saved
        if (not is_last): num_run -= 1 
        for i in range(1, num_run):
            self.run_cycles += max(fst_pipeline_costs[i+1], snd_pipeline_costs[i]) # Decided by the Higher part
        # print (fst_pipeline_costs)
        # print (snd_pipeline_costs)
        # print (trd_pipeline_costs)
        logging.info("Runtime cost of FFT takes %d cycles"%(self.run_cycles - current_cycle))
        logging.info("Total Runtime cost is %d cycles"%(self.run_cycles))
        logging.info("##############")
        return self.run_cycles - current_cycle


    def run_bfly(self, height, width1, width2,is_last=False, ): # Linear projection layer after attention
        current_cycle = self.run_cycles
        logging.info("Running Butterfly layer")
        if width1 < width2: multi_width = int(width2 / width1)
        else: multi_width = 1
        num_run = int(math.ceil(float(height) / self.parallesm_be)) * multi_width
        width = width1
        ############################# First Level Pipelining #############################
        # Get data from dram to bram
        dram_data_read_cycles = self.indata_dram.read(width, self.parallesm_be, self.bit_width) # Read from dram
        for i in range(self.parallesm_be):
            self.data_bram_a[i].write(width, 1, self.bit_width)
            self.data_bram_b[i].write(width, 1, self.bit_width)
        bram_data_write_cycles = width #  Due to Serial to Parallel module, input data comes in one by one
        input_data_cycles = max(dram_data_read_cycles, bram_data_write_cycles) 

        # Get coef from dram to bram
        dram_coef_read_cycles = self.coef_dram.read((width).bit_length()-1, 2*width, self.bit_width) # non-symmetric, Log(N) * 2 * N parameters
        bram_coef_write_cycles = self.coef_bram.write((width).bit_length()-1, 2*width, self.bit_width) # non-Log(N) 2* * N parameters
        weight_data_cycles = max(dram_coef_read_cycles, bram_coef_write_cycles)
        print ("input_data transfer cycles:", input_data_cycles)
        print ("weight_data transfer cycles:", weight_data_cycles)
        print ("dram_coef_read transfer cycles:", dram_coef_read_cycles)
        print ("bram_coef_write transfer cycles:", bram_coef_write_cycles)
        fst_pipeline_cost = max(input_data_cycles, weight_data_cycles) 
        logging.info("First-level Pipeline: Loading data/coef from Dram and fft computetakes %d cycles", fst_pipeline_cost)

        fst_pipeline_costs = [fst_pipeline_cost for i in range(num_run)]
        # Padding
        for i in range(self.bfly_pipeline_stage-1): fst_pipeline_costs.append(0)
        self.run_cycles += fst_pipeline_costs[0]  # Pipeline, obtain max cycles as the real cycle

        # Start to compute FFT
        bram_data_read_cycles = []
        for i in range(self.parallesm_be):
            bram_a_data_read_cycles = 0
            bram_b_data_read_cycles = 0
            for j in range((width).bit_length()-1):
                bram_a_data_read_cycles += self.data_bram_a[i].read(width, 1, self.bit_width)
                bram_b_data_read_cycles += self.data_bram_b[i].read(width, 1, self.bit_width)
            bram_data_read_cycles.append(bram_a_data_read_cycles)
            bram_data_read_cycles.append(bram_b_data_read_cycles)
        bram_data_read_cycles = max(bram_data_read_cycles)

        bfly_compute_time = self.bfly_engines[0].run(width) # Run in parallel, get the first one
        bfly_time = max(bfly_compute_time, bram_data_read_cycles)
        print ("bfly compute cycles:", bfly_time)
        snd_pipeline_cost = bfly_time 
        logging.info("First-level Pipeline: Loading data/coef from Dram and fft computetakes %d cycles", fst_pipeline_cost)

        snd_pipeline_costs = [snd_pipeline_cost for i in range(num_run)]
        # Padding
        for i in range(self.fft_pipeline_stage-2): snd_pipeline_costs.append(0)
        self.run_cycles += snd_pipeline_costs[0]  # Pipeline, obtain max cycles as the real cycle 

        # Output data from bram to dram
        dram_data_write_cycles = self.outdata_dram.write(width, self.parallesm_be, self.bit_width) # Write dram, real + complex
        for i in range(self.parallesm_be):
            self.data_bram_a[i].read(width, 1, self.bit_width)
            self.data_bram_b[i].read(width, 1, self.bit_width)
        bram_data_read_cycles =  width # Due to Parallel to Serial module, output one by one
        output_data_cycles = max(dram_data_write_cycles, bram_data_read_cycles)

        trd_pipeline_costs = [output_data_cycles for i in range(num_run)]
        
        ############################# Calcumulate stage by stage in total #############################
        # if not the last layer, the output can be also overlap with the input of next layer, the time spend on last outputing can be saved
        if (not is_last): num_run -= 1 
        for i in range(1, num_run):
            self.run_cycles += max(fst_pipeline_costs[i+2], snd_pipeline_costs[i+1], trd_pipeline_costs[i]) # Decided by the Higher part
        # print (fst_pipeline_costs)
        # print (snd_pipeline_costs)
        # print (trd_pipeline_costs)
        logging.info("Runtime cost of Butterfly takes %d cycles"%(self.run_cycles - current_cycle))
        logging.info("Total Runtime cost is %d cycles"%(self.run_cycles))
        logging.info("##############")
        return self.run_cycles - current_cycle
