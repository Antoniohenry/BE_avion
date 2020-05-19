#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, math, numpy as np, matplotlib.pyplot as plt

import dynamic as dyn, utils as ut


def plot_thrust(P, filename=None):
    figure = ut.prepare_fig(None, u'PoussÃ©e {}'.format(P.name))
    U = [0, 1., 0, 0]
    hs, machs = np.linspace(3000, 11000, 5), np.linspace(0.5, 0.8, 30)
    for h in hs:
        thrusts = [dyn.propulsion_model([0, h, dyn.va_of_mach(mach, h), 0, 0, 0], U, P) for mach in machs]
        plt.plot(machs, thrusts)
    ut.decorate(plt.gca(), u'PoussÃ©e maximum {}'.format(P.eng_name), 'Mach', '$N$',
                ['{} m'.format(h) for h in hs])
    if filename is not None: plt.savefig(filename, dpi=160)
    return figure


def CL(P, alpha, dphr): return dyn.get_aero_coefs(1, alpha, 0, dphr, P)[0]


def plot_CL(P, filename=None):
    alphas = np.linspace(ut.rad_of_deg(-10), ut.rad_of_deg(20), 30)
    dms = np.linspace(ut.rad_of_deg(20), ut.rad_of_deg(-30), 3)
    figure = ut.prepare_fig(None, u'Coefficient de Portance {}'.format(P.name))
    for dm in dms:
        plt.plot(ut.deg_of_rad(alphas), CL(P, alphas, dm))
    ut.decorate(plt.gca(), u'Coefficient de Portance {}'.format(P.name), r'$\alpha$ en degres', '$C_L$')
    plt.legend(['$\delta _{{PHR}} =  ${:.1f}'.format(ut.deg_of_rad(dm)) for dm in dms], loc='best')
    if filename is not None: plt.savefig(filename, dpi=160)


def Cm(P, alpha):
    Cma = -P.ms*P.CLa
    return P.Cm0 + Cma*(alpha-P.a0)


def plot_Cm(P, filename=None):
    alphas = np.linspace(ut.rad_of_deg(-10), ut.rad_of_deg(20), 30)
    mss = [-0.1, 0., 0.2, 1.]
    figure = ut.prepare_fig(None, u'Coefficient de moment {}'.format(P.name))
    for ms in mss:
        P.set_mass_and_static_margin(0.5, ms)
        plt.plot(ut.deg_of_rad(alphas), Cm(P, alphas))
    ut.decorate(plt.gca(), u'Coefficient de moment {}'.format(P.name), r'$\alpha$ en degres', '$C_m$',
                ['$ms =  ${: .1f}'.format(ms) for ms in mss])
    if filename is not None: plt.savefig(filename, dpi=160)


def seance_1(ac=dyn.Param_A321()):
    plot_thrust(ac, 'plots/{}_thrust.png'.format(ac.get_name()))
    # plot_CL(ac, 'plots/{}_CL.png'.format(ac.get_name()))
    # plot_Cm(ac, 'plots/{}_Cm.png'.format(ac.get_name()))
    #plot_dphr_e(ac, '../plots/{}_dPHR.png'.format(ac.get_name()))
    #plot_CLe(ac, '../plots/{}_CLe.png'.format(ac.get_name()))
    #plot_polar(ac, '../plots/{}_polar.png'.format(ac.get_name()))


if __name__ == "__main__":
    if 'all' in sys.argv:
        for t in dyn.all_ac_types:
            seance_1(t())
    else:
        seance_1(dyn.Param_A320())
        plt.show()