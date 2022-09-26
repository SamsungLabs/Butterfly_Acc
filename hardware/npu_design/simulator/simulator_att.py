from att_accelerator import Att_Accelerator
import argparse
import logging

logger = logging.getLogger()


def simulation(args):
    if args.debug:
        logger.setLevel(logging.DEBUG) 
    else:
        logger.setLevel(logging.INFO)
    if args.version == "base":
        num_layer = 12
        hidden_dim = 768
        ffn_inner_dim = 3072
    elif args.version == "large":
        num_layer = 24
        hidden_dim = 1024
        ffn_inner_dim = 4096
    else:
        raise NotImplementedError("Not supported version.")

    design = Att_Accelerator(args.head_dim, hidden_dim, args.num_len, ffn_inner_dim)
    design.run_att()
    design.run_lp()
    design.run_fc1()
    design.run_fc2()

    network_run_cost = num_layer * design.run_cycles
    ms_per_clock = (1/args.frequency/1000) / args.efficiency
    print ("The overall latecy is:", network_run_cost*ms_per_clock) 
    logging.info("####################Finish######################")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--head_dim", default=64, type=int, help="Dimension per head")
    # parser.add_argument("--hidden_dim", default=128, type=int, help="Hidden dimension")
    parser.add_argument("--num_len", default=64, type=int, help="Lengh of input sequence")
    # parser.add_argument("--ffn_inner_dim", default=512, type=int, help="Inner dimension of FFN")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--version", default="base", type=str, help="base of large")
    parser.add_argument("--frequency", default=200, type=int, help="The frequency of the design")
    parser.add_argument("--efficiency", default=0.85, type=float, help="The hardware implementation efficiency")
    

    args = parser.parse_args()

    simulation(args)