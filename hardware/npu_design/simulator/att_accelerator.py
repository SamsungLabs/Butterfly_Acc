from bram import Bram
from dram import Dram
from compute_engine import CE
import logging
import math

logger = logging.getLogger(__name__)


class Att_Accelerator:
    def __init__(self, head_dim, hidden_dim, num_len, ffn_inner_dim, pr_lt=1, pv_lt=64, pv_ln=8, mac_factor=1, bit_width=16,
                    dram_bw=2048, softmax_delay=72, p_head = 8):
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.ffn_inner_dim = ffn_inner_dim
        self.num_len = num_len
        self.pv_lt = pv_lt # parallelism vector of linear transformation
        self.pr_lt = pr_lt # parallelism row of linear transformation
        self.pc_lt = 1
        self.pv_ln = pv_ln
        self.mac_factor = mac_factor
        self.p_head = self.mac_factor * p_head
        self.bit_width = bit_width
        self.dram_bw = dram_bw
        self.run_cycles = 0
        self.softmax_delay = softmax_delay
        # Determine the parallelsim to fully pipeline the design without stall
        self.pr_qk = self.pr_lt
        self.pc_qk = self.mac_factor
        self.pv_qk = (self.pv_lt / self.pc_qk) / 2
        self.pr_sv = self.pr_qk
        self.pv_sv = self.pv_qk
        self.pc_sv = self.mac_factor
        ############################# Define Dram #############################
        self.dram = Dram(self.dram_bw)

        ############################# Define Bram #############################
        # Data BRAM with size num_len * hidden_dim * bit_with
        # self.data_bram = Bram(self.num_len * self.hidden_dim // (self.pr_lt * self.pv_lt * self.pc_lt), self.bit_width, self.pr_lt * self.pv_lt * self.pc_lt, "data_bram")
        self.data_bram = Bram(self.num_len * self.ffn_inner_dim // (self.pr_lt * self.pv_lt * self.pc_lt), self.bit_width, self.pr_lt * self.pv_lt * self.pc_lt, "data_bram")
        # Coef BRAM with size hidden_dim * p_h * head_dim * bit_with
        # self.coef_bram = Bram(self.hidden_dim * self.head_dim * self.p_head // (self.pr_lt * self.pv_lt * self.pc_lt * self.p_head), self.bit_width, self.pr_lt * self.pv_lt * self.pc_lt * self.p_head, "coef_bram")
        self.coef_bram = Bram(self.hidden_dim * self.ffn_inner_dim // (self.pr_lt * self.pv_lt * self.pc_lt * self.p_head * 4), self.bit_width, self.pr_lt * self.pv_lt * self.pc_lt * self.p_head * 4, "coef_bram")

        self.query_brams = []
        for i in range(self.p_head): 
            # Query BRAM with size num_len * head_dim
            self.query_bram = Bram(self.num_len * self.head_dim // (self.pv_qk * self.pr_qk), self.bit_width, self.pv_qk * self.pr_qk, "query_bram%d"%(i))
            self.query_brams.append(self.query_bram)
        self.key_brams = []
        for i in range(self.p_head):
            # Key BRAM with size num_len * head_dim
            self.key_bram = Bram(self.num_len * self.head_dim // (self.pv_qk * self.pr_qk * self.pc_qk), self.bit_width, self.pv_qk * self.pr_qk * self.pc_qk, "key_bram%d"%(i))
            self.key_brams.append(self.key_bram)
        self.value_brams = []
        for i in range(self.p_head):
            # Value BRAM with size num_len * head_dim
            self.value_bram = Bram(self.num_len * self.head_dim // (self.pv_sv * self.pr_sv * self.pc_sv), self.bit_width, self.pv_sv * self.pr_sv * self.pc_sv, "value_bram%d"%(i)) 
            self.value_brams.append(self.value_bram)
        self.score_brams = []
        for i in range(self.p_head):
            # Score BRAM with size num_len * num_len
            self.score_bram = Bram(self.num_len * self.num_len // (self.pv_sv * self.pr_sv), self.bit_width, self.pv_sv * self.pr_sv, "score_bram%d"%(i)) 
            self.score_brams.append(self.score_bram)
        
        ####################### Define Compute Engine #########################
        self.q_ce = CE(self.pv_lt, self.pr_lt * self.pc_lt * self.p_head, "query_ce")
        self.k_ce = CE(self.pv_lt, self.pr_lt * self.pc_lt * self.p_head, "key_ce")
        self.v_ce = CE(self.pv_lt, self.pr_lt * self.pc_lt * self.p_head, "value_ce")

        self.qk_ces = []
        for i in range(self.p_head):
            self.qk_ce = CE(self.pr_qk * self.pv_qk, self.pc_qk, "qk_ce%d"%(i), reg_initial_delay=1)
            self.qk_ces.append(self.qk_ce)
        self.sv_ces = []
        for i in range(self.p_head):
            self.sv_ce = CE(self.pr_sv * self.pv_sv, self.pc_sv, "sv_ce%d"%(i), reg_initial_delay=1)
            self.sv_ces.append(self.sv_ce)

        logging.info("Linear Transform Parallelism Row %d, Parallelism Colmn %d, Parallelism Vector %d"%(self.pr_lt, self.pc_lt, self.pv_lt))
        logging.info("Query * Key Parallelism Row %d, Parallelism Colmn %d, Parallelism Vector %d"%(self.pr_qk, self.pc_qk, self.pv_qk))
        logging.info("Score * Value Parallelism Row %d, Parallelism Colmn %d, Parallelism Vector %d"%(self.pr_sv, self.pc_sv, self.pv_sv))

    def reset_stat(self):
        self.run_cycles = 0

    def run_att(self):
        logging.info("Running self-attention layer")
        start_cycle = self.run_cycles
        num_run = math.ceil(self.hidden_dim / self.head_dim / self.p_head)
        ############################# First Level Pipelining #############################
        # Get data from dram to bram
        dram_data_read_cycles = self.dram.read(self.num_len, self.hidden_dim, self.bit_width) # Read from dram
        bram_data_write_cycles = self.data_bram.write(self.num_len, self.hidden_dim, self.bit_width) # Write fromo bram
        fst_pipeline_cost = max(dram_data_read_cycles, bram_data_write_cycles) 
        # Get coef from dram to bram
        dram_coef_read_cycles = self.dram.read(self.p_head * self.head_dim, self.hidden_dim, self.bit_width) # Read from dram
        bram_coef_write_cycles = self.coef_bram.write(self.p_head * self.head_dim, self.hidden_dim, self.bit_width) # Write fromo bram
        fst_pipeline_cost += max(dram_coef_read_cycles, bram_coef_write_cycles)  
        logging.info("First-level Pipeline: Loading data/coef from Dram takes %d cycles", fst_pipeline_cost)

        fst_pipeline_costs = [fst_pipeline_cost for i in range(num_run)]
        # Padding
        for i in range(2): fst_pipeline_costs.append(0)
        self.run_cycles += fst_pipeline_costs[0]  # Pipeline, obtain max cycles as the real cycle 

        # Start to compute Linear Transformation
        ############################# Second Level Pipelining #############################
        # Data stay, Coef repeat
        bram_data_read_cycles = self.data_bram.read(self.num_len, self.hidden_dim, self.bit_width, 1, 1) # Read to data bram
        bram_coef_read_cycles = self.coef_bram.read(self.p_head * self.head_dim, self.hidden_dim, self.bit_width, 1, self.num_len)
        initial_compute_delay_query, q_compute_time = self.q_ce.run(self.num_len, self.head_dim * self.p_head, self.hidden_dim) # Heigh, Width
        initial_compute_delay_key, k_compute_time = self.k_ce.run(self.num_len, self.head_dim * self.p_head, self.hidden_dim)
        initial_compute_delay_value, v_compute_time = self.v_ce.run(self.num_len, self.head_dim * self.p_head, self.hidden_dim)

        for i in range(self.p_head):
            bram_query_write_cycles = self.query_brams[i].write(self.num_len, self.head_dim, self.bit_width) # Write to query bram
            bram_key_write_cycles = self.key_brams[i].write(self.num_len, self.head_dim, self.bit_width)
            bram_value_write_cycles = self.value_brams[i].write(self.num_len, self.head_dim, self.bit_width)

        snd_pipeline_cost = max(bram_data_read_cycles, bram_coef_read_cycles,
                                q_compute_time, k_compute_time,
                                bram_query_write_cycles, bram_key_write_cycles) 
        logging.info("Second-level Pipeline: Linear Transformation takes %d cycles", snd_pipeline_cost)
        logging.info("Initial Linear Transformation (second-level pipeline) takes %d cycles", initial_compute_delay_value)

        snd_pipeline_costs = [snd_pipeline_cost for i in range(num_run)]
        # Padding
        for i in range(1):
            snd_pipeline_costs.append(0)
        snd_pipeline_costs[0] += initial_compute_delay_query
        self.run_cycles += max(fst_pipeline_costs[1], snd_pipeline_costs[0]) # Pipeline, obtain max cycles as the real cycle

        ############################# Third Level Pipelining #############################
        # Start to compute Query * Keys, row-wise pipeline
        qk_cycles = []
        for i in range(self.num_len):
            for i in range(self.p_head):
                bram_query_read_cycles_per_row = self.query_brams[i].read(1, self.head_dim, self.bit_width, self.num_len / self.pc_qk, 1)
                bram_key_read_cycles_per_row = self.key_brams[i].read(self.num_len, self.head_dim, self.bit_width, 1, 1)
            initial_compute_delay_score, score_compute_time_per_row = self.qk_ce.run(1, self.num_len, self.head_dim)
            qk_cycles.append(max(bram_query_read_cycles_per_row, bram_key_read_cycles_per_row,
                                score_compute_time_per_row))# Pipeline, obtain max cycles as the real cycle
        initial_2in1_delay = 3
        initial_smart_mem_delay = 1
        initial_qk_delay = initial_compute_delay_score + initial_2in1_delay + initial_smart_mem_delay
        trd_pipeline_cost = qk_cycles[0] + initial_qk_delay
        logging.info("Initial QK in Self Attention (third-level pipeline) takes %d + %d cycles"%(qk_cycles[0], initial_qk_delay))

        # Start to compute softmax        
        softmax_cycles = []
        initial_softmax_delay = self.softmax_delay
        for i in range(self.num_len):
            softmax_time = self.num_len
            for i in range(self.p_head):
                bram_score_write_cycles_per_row = self.score_brams[i].write(1, self.num_len, self.bit_width)
            softmax_cycles.append(max(softmax_time, bram_score_write_cycles_per_row))
        trd_pipeline_cost += max(qk_cycles[1], softmax_cycles[0] + initial_softmax_delay)
        logging.info("Initial Softmax in Self Attention (third-level pipeline) takes %d cycles", max(qk_cycles[1], softmax_cycles[0] + initial_softmax_delay))

        # Start to compute Score * Value
        sv_cycles = []
        for i in range(self.num_len):
            for i in range(self.p_head):
                bram_score_read_cycles_per_row = self.score_brams[i].read(1, self.num_len, self.bit_width, self.head_dim / self.pc_sv, 1) #Score stay, value repeat
                bram_value_read_cycles_per_row = self.value_brams[i].read(self.head_dim, self.num_len, self.bit_width, 1, 1)
            initial_compute_delay_sv, sv_compute_time_per_row = self.sv_ce.run(1, self.head_dim, self.num_len)
            # Write back to Dram directly
            sv_cycles.append(max(bram_score_read_cycles_per_row, bram_value_read_cycles_per_row, sv_compute_time_per_row))
        trd_pipeline_cost += initial_compute_delay_sv
        logging.info("Initial SV in Self Attention (third-level pipeline) takes %d cycles", initial_compute_delay_sv)

        # Calculate the runtime row by row in third stage
        for i in range(2, self.num_len):
            trd_pipeline_cost += max(qk_cycles[i], softmax_cycles[i-1], sv_cycles[i-2])
            #self.run_cycles += max(qk_cycles[i], softmax_cycles[i-1], sv_cycles[i-2])# Pipeline, obtain max cycles as the real cycle
        trd_pipeline_cost += max(softmax_cycles[self.num_len-1], sv_cycles[self.num_len-2])
        trd_pipeline_cost += sv_cycles[self.num_len-1]
        trd_pipeline_costs = [trd_pipeline_cost]
        for i in range(num_run-1):
            trd_pipeline_costs.append(self.num_len * max(qk_cycles[-1], softmax_cycles[-1], sv_cycles[-1]))

        self.run_cycles += max(fst_pipeline_costs[2], snd_pipeline_costs[1], trd_pipeline_costs[0])
        logging.info("Second-level Pipeline: Self Attention takes %d cycles", trd_pipeline_costs[0])
        
        ############################# Calcumulate stage by stage in total #############################

        for i in range(1, num_run):
            self.run_cycles += max(fst_pipeline_costs[i+2], snd_pipeline_costs[i+1], trd_pipeline_costs[i])
        # print (fst_pipeline_costs)
        # print (snd_pipeline_costs)
        # print (trd_pipeline_costs)
        logging.info("Runtime cost of self attention takes %d cycles"%(self.run_cycles-start_cycle))
        return self.run_cycles-start_cycle



    def run_lp(self): # Linear projection layer after attention
        logging.info("Running linear projection layer")
        start_cycle = self.run_cycles
        ############################# First Level Pipelining #############################
        # Get data from dram to bram
        dram_data_read_cycles = self.dram.read(self.num_len, self.hidden_dim, self.bit_width) # Read from dram
        bram_data_write_cycles = self.data_bram.write(self.num_len, self.hidden_dim, self.bit_width) # Write fromo bram
        # print (dram_data_read_cycles, bram_data_write_cycles)
        fst_pipeline_cost = max(dram_data_read_cycles, bram_data_write_cycles) 
        # Get coef from dram to bram
        dram_coef_read_cycles = self.dram.read(self.hidden_dim, self.hidden_dim, self.bit_width) # Read from dram
        bram_coef_write_cycles = self.coef_bram.write(self.hidden_dim, self.hidden_dim, self.bit_width) # Write fromo bram
        # print (dram_coef_read_cycles, bram_coef_write_cycles)
        fst_pipeline_cost += max(dram_coef_read_cycles, bram_coef_write_cycles)  
        logging.info("First-level Pipeline: Loading data/coef from Dram takes %d cycles", fst_pipeline_cost)
        self.run_cycles += fst_pipeline_cost

        # Start to compute the linear projection layer
        ############################# Second Level Pipelining #############################
        # Data stay, Coef repeat
        for i in range(self.num_len):
            bram_data_read_cycles = self.data_bram.read(1, self.hidden_dim, self.bit_width, 1, 1) # Read to data bram
            bram_coef_read_cycles = self.coef_bram.read(self.hidden_dim, self.hidden_dim, self.bit_width, 1, 1)
            initial_compute_delay_fc_qeury_ce, fc_query_ce_compute_time = self.q_ce.run(1, self.hidden_dim/4, self.hidden_dim) # Heigh, Width
            initial_compute_delay_fc_key_ce, fc_key_ce_compute_time = self.k_ce.run(1, self.hidden_dim/4, self.hidden_dim) # Heigh, Width
            initial_compute_delay_fc_value_ce, fc_value_ce_compute_time = self.v_ce.run(1, self.hidden_dim/4, self.hidden_dim) # Heigh, Width
            initial_compute_delay_fc_qk_ce, fc_qk_ce_compute_time = self.qk_ce.run(1, self.hidden_dim/8/self.p_head, self.hidden_dim) # Heigh, Width, There are p_head number of ce, so divide it by p_head
            initial_compute_delay_fc_sv_ce, fc_sv_ce_compute_time = self.sv_ce.run(1, self.hidden_dim/8/self.p_head, self.hidden_dim) # Heigh, Width, There are p_head number of ce, so divide it by p_head

        initial_compute_delay_fc = max(initial_compute_delay_fc_qeury_ce, initial_compute_delay_fc_key_ce, initial_compute_delay_fc_value_ce,
                                        initial_compute_delay_fc_qk_ce, initial_compute_delay_fc_sv_ce)
        fc_compute_time = max(fc_query_ce_compute_time, fc_key_ce_compute_time, fc_value_ce_compute_time,
                                    fc_qk_ce_compute_time, fc_sv_ce_compute_time)
        # print (fc_query_ce_compute_time, fc_key_ce_compute_time, fc_value_ce_compute_time,
        #                             fc_qk_ce_compute_time, fc_sv_ce_compute_time)
        snd_pipeline_cost = max(bram_data_read_cycles, bram_coef_read_cycles, fc_compute_time)
        # print (bram_data_read_cycles, bram_coef_read_cycles, fc_compute_time)
        self.run_cycles += initial_compute_delay_fc + snd_pipeline_cost
        snd_pipeline_costs = [snd_pipeline_cost for i in range(self.num_len)]

        logging.info("Second-level Pipeline: Linear Projection takes %d cycles", snd_pipeline_cost)
        logging.info("Initial Linear Projection (second-level pipeline) takes %d cycles", initial_compute_delay_fc)

        ############################# Third Level Pipelining #############################
        # Calculate the initial delay of LN
        mean_delay = math.log(self.pv_ln) + (self.hidden_dim//self.pv_ln) * (self.hidden_dim//self.pv_lt) + 2
        fanout_delay = 4
        sub_delay = 2 # sub + reg
        square_delay = 3 # mult + reg
        root_delay = 16
        div_delay = 68
        var_delay = sub_delay + square_delay + math.log(self.pv_ln) + (self.hidden_dim//self.pv_ln) + root_delay + div_delay
        trd_pipeline_cost = mean_delay + var_delay # initial cost
        self.run_cycles += max(snd_pipeline_costs[0], trd_pipeline_cost)
        trd_pipeline_costs = [(self.hidden_dim//self.pv_ln) * (self.hidden_dim//self.pv_lt) for i in range(self.num_len)]
        logging.info("Third-level Pipeline: Linear Normalization takes %d cycles", trd_pipeline_cost)

        forth_pipeline_cost = self.hidden_dim//self.pv_ln # initial cost
        self.run_cycles += max(snd_pipeline_costs[1], trd_pipeline_costs[0], forth_pipeline_cost)
        forth_pipeline_costs = [self.hidden_dim//self.pv_ln for i in range(self.num_len)]
        # Simpley add together as that is a single run

        for i in range(self.num_len-2):
            self.run_cycles += max(snd_pipeline_costs[i+2], trd_pipeline_costs[i+1], forth_pipeline_costs[i])
        # print (snd_pipeline_costs)
        # print (trd_pipeline_costs)
        # print (forth_pipeline_costs)
        logging.info("Runtime cost of linear projection takes %d cycles"%(self.run_cycles - start_cycle))
        return self.run_cycles-start_cycle
        

    # Run the first FC layer in FFN
    def run_fc1(self):
        logging.info("Running the first fc layer in FFN")
        start_cycle = self.run_cycles
        ############################# First Level Pipelining #############################
        # Get data from dram to bram
        dram_data_read_cycles = self.dram.read(self.num_len, self.hidden_dim, self.bit_width) # Read from dram
        bram_data_write_cycles = self.data_bram.write(self.num_len, self.hidden_dim, self.bit_width) # Write fromo bram
        fst_pipeline_cost = max(dram_data_read_cycles, bram_data_write_cycles) 
        # Get coef from dram to bram
        dram_coef_read_cycles = self.dram.read(self.ffn_inner_dim, self.hidden_dim, self.bit_width) # Read from dram
        bram_coef_write_cycles = self.coef_bram.write(self.ffn_inner_dim, self.hidden_dim, self.bit_width) # Write fromo bram
        fst_pipeline_cost += max(dram_coef_read_cycles, bram_coef_write_cycles)  
        logging.info("First-level Pipeline: Loading data/coef from Dram takes %d cycles", fst_pipeline_cost)
 
        # Start to compute the first FC layer
        ############################# Second Level Pipelining #############################
        # Data stay, Coef repeat
        for i in range(self.num_len):
            bram_data_read_cycles = self.data_bram.read(1, self.hidden_dim, self.bit_width, 1, 1) # Read to data bram
            bram_coef_read_cycles = self.coef_bram.read(self.ffn_inner_dim, self.hidden_dim, self.bit_width, 1, 1)
            initial_compute_delay_fc_qeury_ce, fc_query_ce_compute_time = self.q_ce.run(1, self.ffn_inner_dim/4, self.hidden_dim) # Heigh, Width
            initial_compute_delay_fc_key_ce, fc_key_ce_compute_time = self.k_ce.run(1, self.ffn_inner_dim/4, self.hidden_dim) # Heigh, Width
            initial_compute_delay_fc_value_ce, fc_value_ce_compute_time = self.v_ce.run(1, self.ffn_inner_dim/4, self.hidden_dim) # Heigh, Width
            initial_compute_delay_fc_qk_ce, fc_qk_ce_compute_time = self.qk_ce.run(1, self.ffn_inner_dim/8/self.p_head, self.hidden_dim) # Heigh, Width
            initial_compute_delay_fc_sv_ce, fc_sv_ce_compute_time = self.sv_ce.run(1, self.ffn_inner_dim/8/self.p_head, self.hidden_dim) # Heigh, Width

        initial_compute_delay_fc = max(initial_compute_delay_fc_qeury_ce, initial_compute_delay_fc_key_ce, initial_compute_delay_fc_value_ce,
                                        initial_compute_delay_fc_qk_ce, initial_compute_delay_fc_sv_ce)
        fc_compute_time = max(fc_query_ce_compute_time, fc_key_ce_compute_time, fc_value_ce_compute_time,
                                fc_qk_ce_compute_time, fc_sv_ce_compute_time)
        snd_pipeline_cost = max(bram_data_read_cycles, bram_coef_read_cycles, fc_compute_time)

        logging.info("Second-level Pipeline: FC1 takes %d cycles", snd_pipeline_cost)
        logging.info("Initial FC1 (second-level pipeline) takes %d cycles", initial_compute_delay_fc)

        # Simpley add together as that is a single run
        self.run_cycles += fst_pipeline_cost
        for i in range(self.num_len):
            self.run_cycles += snd_pipeline_cost
        logging.info("Runtime cost of the first FC in FFN takes %d cycles"%(self.run_cycles - start_cycle))
        return self.run_cycles-start_cycle


    def run_fc2(self): # Run the second FC layer in FFN
        logging.info("Running the second fc layer in FFN")
        start_cycle = self.run_cycles
        ############################# First Level Pipelining #############################
        # Get data from dram to bram
        dram_data_read_cycles = self.dram.read(self.num_len, self.ffn_inner_dim, self.bit_width) # Read from dram
        bram_data_write_cycles = self.data_bram.write(self.num_len, self.ffn_inner_dim, self.bit_width) # Write fromo bram
        fst_pipeline_cost = max(dram_data_read_cycles, bram_data_write_cycles) 
        # Get coef from dram to bram
        dram_coef_read_cycles = self.dram.read(self.hidden_dim, self.ffn_inner_dim, self.bit_width) # Read from dram
        bram_coef_write_cycles = self.coef_bram.write(self.hidden_dim, self.ffn_inner_dim, self.bit_width) # Write fromo bram
        fst_pipeline_cost += max(dram_coef_read_cycles, bram_coef_write_cycles)  
        logging.info("First-level Pipeline: Loading data/coef from Dram takes %d cycles", fst_pipeline_cost)
 
        # Start to compute the linear projection layer
        ############################# Second Level Pipelining #############################
        # Data stay, Coef repeat
        for i in range(self.num_len):
            bram_data_read_cycles = self.data_bram.read(1, self.ffn_inner_dim, self.bit_width, 1, 1) # Read to data bram
            bram_coef_read_cycles = self.coef_bram.read(self.hidden_dim, self.ffn_inner_dim, self.bit_width, 1, 1)
            initial_compute_delay_fc_qeury_ce, fc_query_ce_compute_time = self.q_ce.run(1, self.hidden_dim/4, self.ffn_inner_dim) # Heigh, Width
            initial_compute_delay_fc_key_ce, fc_key_ce_compute_time = self.k_ce.run(1, self.hidden_dim/4, self.ffn_inner_dim) # Heigh, Width
            initial_compute_delay_fc_value_ce, fc_value_ce_compute_time = self.v_ce.run(1, self.hidden_dim/4, self.ffn_inner_dim) # Heigh, Width
            initial_compute_delay_fc_qk_ce, fc_qk_ce_compute_time = self.qk_ce.run(1, self.hidden_dim/8/self.p_head, self.ffn_inner_dim) # Heigh, Width
            initial_compute_delay_fc_sv_ce, fc_sv_ce_compute_time = self.sv_ce.run(1, self.hidden_dim/8/self.p_head, self.ffn_inner_dim) # Heigh, Width

        initial_compute_delay_fc = max(initial_compute_delay_fc_qeury_ce, initial_compute_delay_fc_key_ce, initial_compute_delay_fc_value_ce,
                                        initial_compute_delay_fc_qk_ce, initial_compute_delay_fc_sv_ce)
        fc_compute_time = max(fc_query_ce_compute_time, fc_key_ce_compute_time, fc_value_ce_compute_time,
                                    fc_qk_ce_compute_time, fc_sv_ce_compute_time)
        snd_pipeline_cost = max(bram_data_read_cycles, bram_coef_read_cycles, fc_compute_time)
        # print ((bram_data_read_cycles, bram_coef_read_cycles, fc_compute_time))
        self.run_cycles += initial_compute_delay_fc + snd_pipeline_cost
        snd_pipeline_costs = [snd_pipeline_cost for i in range(self.num_len)]

        logging.info("Second-level Pipeline: FC2 takes %d cycles", snd_pipeline_cost)
        logging.info("Initial  FC2 (second-level pipeline) takes %d cycles", initial_compute_delay_fc)

        ############################# Third Level Pipelining #############################
        # Calculate the initial delay of LN
        mean_delay = math.log(self.pv_ln) + (self.hidden_dim//self.pv_ln) * (self.hidden_dim//self.pv_lt) + 2
        fanout_delay = 4
        sub_delay = 2 # sub + reg
        square_delay = 3 # mult + reg
        root_delay = 16
        div_delay = 68
        var_delay = sub_delay + square_delay + math.log(self.pv_ln) + (self.hidden_dim//self.pv_ln) + root_delay + div_delay
        trd_pipeline_cost = mean_delay + var_delay # initial cost
        self.run_cycles += max(snd_pipeline_costs[0], trd_pipeline_cost)
        trd_pipeline_costs = [(self.hidden_dim//self.pv_ln) * (self.hidden_dim//self.pv_lt) for i in range(self.num_len)]

        forth_pipeline_cost = self.hidden_dim//self.pv_ln # initial cost
        self.run_cycles += max(snd_pipeline_costs[1], trd_pipeline_costs[0], forth_pipeline_cost)
        forth_pipeline_costs = [self.hidden_dim//self.pv_ln for i in range(self.num_len)]
        # Simpley add together as that is a single run

        for i in range(self.num_len-2):
            self.run_cycles += max(snd_pipeline_costs[i+2], trd_pipeline_costs[i+1], forth_pipeline_costs[i])

        # print (snd_pipeline_costs)
        # print (trd_pipeline_costs)
        # print (forth_pipeline_costs)

        logging.info("Runtime cost of the first FC in FFN takes %d cycles"%(self.run_cycles - start_cycle))
        return self.run_cycles-start_cycle


    def run_fft(self, complex_input=False, complex_output=True):
        logging.info("Running FFT on baseline design, equivalent ot running FC layer")
        start_cycle = self.run_cycles
        ############################# First Level Pipelining #############################
        # Get data from dram to bram
        if (complex_input): dram_data_read_cycles = self.dram.read(self.num_len, self.hidden_dim, self.bit_width*2) # Read from dram
        else: dram_data_read_cycles = self.dram.read(self.num_len, self.hidden_dim, self.bit_width*2) # Read from dram
        if (complex_output): bram_data_write_cycles = self.data_bram.write(self.num_len, self.hidden_dim, self.bit_width*2) # Write fromo bram
        else: bram_data_write_cycles = self.data_bram.write(self.num_len, self.hidden_dim, self.bit_width) # Write fromo bram
        fst_pipeline_cost = max(dram_data_read_cycles, bram_data_write_cycles) 
        # Get coef from dram to bram
        dram_coef_read_cycles = self.dram.read(self.hidden_dim, self.hidden_dim, self.bit_width) # Read from dram
        bram_coef_write_cycles = self.coef_bram.write(self.hidden_dim, self.hidden_dim, self.bit_width) # Write fromo bram
        fst_pipeline_cost += max(dram_coef_read_cycles, bram_coef_write_cycles)  
        logging.info("First-level Pipeline: Loading data/coef from Dram takes %d cycles", fst_pipeline_cost)
 
        # Start to compute the first FC layer
        ############################# Second Level Pipelining #############################
        # Data stay, Coef repeat
        for i in range(self.num_len):
            bram_data_read_cycles = self.data_bram.read(1, self.hidden_dim, self.bit_width, 1, 1) # Read to data bram
            bram_coef_read_cycles = self.coef_bram.read(self.hidden_dim, self.hidden_dim, self.bit_width, 1, 1)
            initial_compute_delay_fc_qeury_ce, fc_query_ce_compute_time = self.q_ce.run(1, self.hidden_dim/4, self.hidden_dim) # Heigh, Width
            initial_compute_delay_fc_key_ce, fc_key_ce_compute_time = self.k_ce.run(1, self.hidden_dim/4, self.hidden_dim) # Heigh, Width
            initial_compute_delay_fc_value_ce, fc_value_ce_compute_time = self.v_ce.run(1, self.hidden_dim/4, self.hidden_dim) # Heigh, Width
            initial_compute_delay_fc_qk_ce, fc_qk_ce_compute_time = self.qk_ce.run(1, self.hidden_dim/8/self.p_head, self.hidden_dim) # Heigh, Width
            initial_compute_delay_fc_sv_ce, fc_sv_ce_compute_time = self.sv_ce.run(1, self.hidden_dim/8/self.p_head, self.hidden_dim) # Heigh, Width

        initial_compute_delay_fc = max(initial_compute_delay_fc_qeury_ce, initial_compute_delay_fc_key_ce, initial_compute_delay_fc_value_ce,
                                        initial_compute_delay_fc_qk_ce, initial_compute_delay_fc_sv_ce)
        fc_compute_time = max(fc_query_ce_compute_time, fc_key_ce_compute_time, fc_value_ce_compute_time,
                                fc_qk_ce_compute_time, fc_sv_ce_compute_time)
        snd_pipeline_cost = max(bram_data_read_cycles, bram_coef_read_cycles, fc_compute_time)

        logging.info("Second-level Pipeline: FC1 takes %d cycles", snd_pipeline_cost)
        logging.info("Initial FC1 (second-level pipeline) takes %d cycles", initial_compute_delay_fc)

        # Simpley add together as that is a single run
        self.run_cycles += fst_pipeline_cost
        for i in range(self.num_len):
            self.run_cycles += snd_pipeline_cost
        logging.info("Runtime cost of the FFT on baseline design takes %d cycles"%(self.run_cycles - start_cycle))
        return self.run_cycles-start_cycle