import argparse
from polyfit import *


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default=1, help="Part of program")
    parser.add_argument("--method", default="pinv", help="type of solver")
    parser.add_argument("--batch_size", default=5, type=int, help="batch size")
    parser.add_argument("--lamb", default=0, type=float,
                        help="regularization constant")
    parser.add_argument("--polynomial", default=10,
                        type=int, help="degree of polynomial")
    parser.add_argument("--result_dir", default="",
                        type=str, help="Files to store plots")
    parser.add_argument("--X", default="", type=str,
                        help="Read content from the file")

    return parser.parse_args()


if __name__ == '__main__':
    args = setup()
    args.method = ('mgrad')if(args.method == 'gd')else ('piv')
    # print(args)
    # load data
    data = np.genfromtxt(args.X, delimiter=',')
    X, Y = data[:, 0], data[:, 1]
    X, Y = zip(*sorted(zip(X, Y)))
    X, Y = np.array(X), np.array(Y)

    # Fit the model
    model = polyfit(degree=args.polynomial, lmda=args.lamb,
                    method=args.method, batch_size=args.batch_size, max_iter=5000, steps=1e-3)
    model.fit(X, Y)
    w = np.reshape(model.w, (-1))
    print(f"weights={w}")
