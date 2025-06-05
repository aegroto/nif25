import torch
import argparse
from matplotlib import pyplot as plt
from serialization import deserialize_state_dict 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state_dict_path")
    parser.add_argument("dump_path")
    args = parser.parse_args()

    state_dict = deserialize_state_dict(args.state_dict_path)
    # state_dict = torch.load(args.state_dict_path)
    for (key, tensor) in state_dict.items():
        try:
            print(key)
            values = tensor # .cpu()# .numpy().flatten()
            # values = values.mul(32).round().div(32)
            # values = tensor - values
            # scale = original_values.var().div(2.0).sqrt()
            # coded_values = laplace_icdf(original_values, scale)
            # bound = values.abs().max()
            # values = values.div(bound)
            # values = torch.log(values)

            # residual_map, samples = generate_residual_prediction_map(values, residual_delta)
            # residual_map = residual_map.cpu().numpy().flatten()
            # plt.clf()
            # # plt.plot(numpy.arange(0, 255), counts)
            # plt.plot(samples.cpu(), residual_map)
            # plt.savefig(f"{args.dump_path}/{key}.resmap.png")

            values = values.cpu().numpy().flatten()
            # symbols, counts = numpy.unique(values, return_counts=True)
            plt.clf()
            # plt.plot(numpy.arange(0, 255), counts)
            plt.hist(values, 255)
            plt.savefig(f"{args.dump_path}/{key}.values.png")

            # residual_delta = residual_delta.cpu().numpy().flatten()
            # plt.clf()
            # plt.hist(residual_delta, 255)
            # plt.savefig(f"{args.dump_path}/{key}.noise.png")

        except Exception as e:
            print(f"{key}: {e}")

if __name__ == "__main__":
    main()
