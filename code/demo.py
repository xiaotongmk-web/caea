"""A small deterministic demo; it neither queries nor trains a language model."""
import argparse
import numpy as np
from scipy.special import expit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pressure',type=float,default=4.)
    parser.add_argument('--repair',type=float,default=.6)
    args=parser.parse_args()
    if args.pressure<0 or not 0<=args.repair<=1:
        parser.error('Use nonnegative pressure and a repair fraction in [0, 1]')
    u=np.array([.47,.53]);eta=.25
    multiplier=1-eta+eta*(1-args.repair)*args.pressure/2
    print('Pooled P(Z=z,Q=q)=0.25 for every binary pair; Brier regret=0.25')
    print(f'Central mean-response derivative: {multiplier:.6f}')
    print('Strict global contraction:',(1-args.repair)*args.pressure<2)
    for t in range(401):
        if t in [0,10,100,400]:
            print(f'Round {t:3d}: beliefs {u[0]:.6f}, {u[1]:.6f}')
        u=(1-eta)*u+eta*((1-args.repair)*expit(args.pressure*(2*u-1))+.5*args.repair)
    print('This is mean-response simulation, not measured human behavior.')


if __name__=='__main__':main()
