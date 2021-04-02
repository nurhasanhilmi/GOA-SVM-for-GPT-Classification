from methods.goa import BaseGOA


def main():
    def obj_func(solution):
        x=solution
        return x[0]**2 + x[1]**2

    verbose = True
    lb = [0, 1]
    ub = [5, 4]
    pop_size = 30
    epoch = 10
    # c_minmax=(0.000001, 5)
    c_minmax=(0.00004, 2)
    md = BaseGOA(obj_func=obj_func, lb=lb, ub=ub, verbose=verbose, pop_size=pop_size, epoch=epoch, c_minmax=c_minmax)
    md.train()


if __name__ == '__main__':
    main()