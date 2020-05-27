#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

import dynamic as dyn
import utils as ut

alphas = np.linspace(ut.rad_of_deg(-10), ut.rad_of_deg(20), 30)


def plot_thrust(P, filename=None):
    """P est le paramètre avion"""
    figure = ut.prepare_fig(None, u'Poussée {}'.format(P.name))
    U = [0, 1., 0, 0]
    hs, machs = np.linspace(3000, 11000, 5), np.linspace(0.5, 0.8, 30)
    for h in hs:
        thrusts = [dyn.propulsion_model([0, h, dyn.va_of_mach(mach, h), 0, 0, 0], U, P) for mach in machs]
        plt.plot(machs, thrusts)
    ut.decorate(plt.gca(), u'Poussée maximum {}'.format(P.eng_name), 'Mach', '$N$',
                ['{} m'.format(h) for h in hs])
    if filename is not None:
        plt.savefig(filename, dpi=160)
    return figure


def CL(P, alpha, dphr):
    return dyn.get_aero_coefs(1, alpha, 0, dphr, P)[0]


def plot_CL(P, filename=None):
    dms = np.linspace(ut.rad_of_deg(20), ut.rad_of_deg(-30), 3)
    figure = ut.prepare_fig(None, u'Coefficient de Portance {}'.format(P.name))
    for dm in dms:
        plt.plot(ut.deg_of_rad(alphas), CL(P, alphas, dm))
    ut.decorate(plt.gca(), u'Coefficient de Portance {}'.format(P.name), r'$\alpha$ en degres', '$C_L$')
    plt.legend(['$\delta _{{PHR}} =  ${:.1f}'.format(ut.deg_of_rad(dm)) for dm in dms], loc='best')
    if filename is not None:
        plt.savefig(filename, dpi=160)
    return figure


def Cm(P, alpha):
    Cma = -P.ms * P.CLa
    return P.Cm0 + Cma * (alpha - P.a0)


def plot_Cm(P, filename=None):
    mss = [-0.1, 0., 0.2, 1.]
    figure = ut.prepare_fig(None, u'Coefficient de moment {}'.format(P.name))
    for ms in mss:
        P.set_mass_and_static_margin(0.5, ms)
        plt.plot(ut.deg_of_rad(alphas), Cm(P, alphas))
    ut.decorate(plt.gca(), u'Coefficient de moment {}'.format(P.name), r'$\alpha$ en degres', '$C_m$',
                ['$ms =  ${: .1f}'.format(ms) for ms in mss])
    if filename is not None:
        plt.savefig(filename, dpi=160)
    return figure


def plot_dphr_e_vt(ac, filename=None):
    ac.Vt = ac.Vt * 1.5
    mss = [-0.1, 0., 0.2, 1.]

    def dphr(alpha, ms):
        return (- ac.Cm0 + ms * ac.CLa * (alpha - ac.a0)) / ac.Cmd

    figure = ut.prepare_fig(None, r'$\delta_{phr}$ en fonction de $\alpha$ avec $V_{t} + 50\%$ {name}'.format(phr='phr',
                                                                                                              t='t',
                                                                                                              name=ac.name))
    for ms in mss:
        dphrs = [dphr(alpha, ms) for alpha in alphas]
        plt.plot(ut.deg_of_rad(alphas), dphrs)
        ut.decorate(plt.gca(), r'$\delta_{phr}$ en fonction de $\alpha$, $V_{t} + 50\%$', r'$\alpha$ en degres',
                    u'$\delta_{phr}$',
                    ['$ms =  ${: .1f}'.format(ms) for ms in mss])
    if filename is not None:
        plt.savefig(filename, dpi=160)
    return figure


def plot_dphr_e(ac, filename=None):
    mss = [-0.1, 0., 0.2, 1.]

    def dphr(alpha, ms):
        return (- ac.Cm0 + ms * ac.CLa * (alpha - ac.a0)) / ac.Cmd

    figure = ut.prepare_fig(None, r'$\delta_{phr}$ en fonction de $\alpha$ {name}'.format(
        phr='phr', t='t', name=ac.name))
    for ms in mss:
        dphrs = [dphr(alpha, ms) for alpha in alphas]
        plt.plot(ut.deg_of_rad(alphas), dphrs)
        ut.decorate(plt.gca(), r'$\delta_{phr}$ en fonction de $\alpha$', r'$\alpha$ en degres', u'$\delta_{phr}$',
                    ['$ms =  ${: .1f}'.format(ms) for ms in mss])
    if filename is not None:
        plt.savefig(filename, dpi=160)
    return figure


def plot_CLe(ac, filename=None):
    mss = [0.2, 1.]

    def dphr(alpha, ms):
        return (- ac.Cm0 + ms * ac.CLa * (alpha - ac.a0)) / ac.Cmd

    def CLe(alpha, ms):
        return dyn.get_aero_coefs(1, alpha, 0, dphr(alpha, ms), ac)[0]

    figure = ut.prepare_fig(None, r'$\delta_{phr}$ en fonction de $\alpha$ {name}'.format(
        phr='phr', t='t', name=ac.name))

    for ms in mss:
        CLes = [CLe(alpha, ms) for alpha in alphas]
        plt.plot(ut.deg_of_rad(alphas), CLes)
        ut.decorate(plt.gca(), r'$C_{Le}$ en fonction de $\alpha$', r'$\alpha$ en degres', u'$C_{Le}$',
                    ['$ms =  ${: .1f}'.format(ms) for ms in mss])

    if filename is not None:
        plt.savefig(filename, dpi=160)
    return figure


def plot_polar(ac, filename=None):
    mss = [0.2, 1.]

    def dphr(alpha, ms):
        return (- ac.Cm0 + ms * ac.CLa * (alpha - ac.a0)) / ac.Cmd

    def CLe(alpha, ms):
        return dyn.get_aero_coefs(1, alpha, 0, dphr(alpha, ms), ac)[0]

    def CDe(alpha, ms):
        return dyn.get_aero_coefs(1, alpha, 0, dphr(alpha, ms), ac)[1]

    figure = ut.prepare_fig(None, r'$\Polaires équilibrées {name}'.format(name=ac.name))

    finesses = []
    for ms in mss:
        CLes = [CLe(alpha, ms) for alpha in alphas]
        CDes = [CDe(alpha, ms) for alpha in alphas]

        pentes = np.array(CLes) / np.array(CDes)
        finesse = round(max(pentes), 3)
        finesses.append(finesse)
        print('finesse : {} pour ms = {}'.format(finesse, ms))

        plt.plot(CDes, CLes)
        ut.decorate(plt.gca(), 'Polaires équilibrées', r'$C_{De}$', u'$C_{Le}$',
                    ['$ms =  ${: .1f}, finesse : {}'.format(ms, finesse) for ms, finesse in zip(mss, finesses)])

    if filename is not None:
        plt.savefig(filename, dpi=160)
    return figure


def seance_1(ac=dyn.Param_A321()):
    # plot_thrust(ac, 'plots/{}_thrust.png'.format(ac.get_name()))
    # plot_CL(ac, 'plots/{}_CL.png'.format(ac.get_name()))
    # plot_Cm(ac, 'plots/{}_Cm.png'.format(ac.get_name()))
    # plot_dphr_e(ac, 'plots/{}_dPHR.png'.format(ac.get_name()))
    # plot_dphr_e_vt(ac, 'plots/{}_dPHR vt augmenter de 50%.png'.format(ac.get_name()))
    # plot_CLe(ac, 'plots/{}_CLe.png'.format(ac.get_name()))
    plot_polar(ac, 'plots/{}_polar.png'.format(ac.get_name()))


if __name__ == "__main__":
    # if 'all' in sys.argv:
    #     for t in dyn.all_ac_types:
    #         seance_1(t())
    # else:
    #     seance_1(dyn.Param_A321())
    #     plt.show()

    seance_1()
