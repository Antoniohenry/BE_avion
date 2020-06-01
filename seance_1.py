#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

import dynamic as dyn
import utils as ut

alphas = np.linspace(ut.rad_of_deg(-10), ut.rad_of_deg(20), 30)


def plot_thrust(aircraft, filename=None):
    """aircraft est le paramètre avion"""
    figure = ut.prepare_fig(None, u'Poussée {}'.format(aircraft.name))
    U = [0, 1., 0, 0]
    hs, machs = np.linspace(3000, 11000, 5), np.linspace(0.5, 0.8, 30)
    for h in hs:
        thrusts = [dyn.propulsion_model([0, h, dyn.va_of_mach(mach, h), 0, 0, 0], U, aircraft) for mach in machs]
        plt.plot(machs, thrusts)
    ut.decorate(plt.gca(), u'Poussée maximum pour un moteur {}'.format(aircraft.eng_name), 'Mach', '$N$',
                ['{} m'.format(h) for h in hs])
    if filename is not None:
        plt.savefig(filename, dpi=160)
    return figure


def CL(aircraft, alpha, dphr):
    return dyn.get_aero_coefs(1, alpha, 0, dphr, aircraft)[0]


def plot_CL(aircraft, filename=None):
    dms = np.linspace(ut.rad_of_deg(20), ut.rad_of_deg(-30), 3)
    figure = ut.prepare_fig(None, u'Coefficient de Portance {}'.format(aircraft.name))
    for dm in dms:
        plt.plot(ut.deg_of_rad(alphas), CL(aircraft, alphas, dm))
    ut.decorate(plt.gca(), u'Coefficient de Portance {}'.format(aircraft.name), r'$\alpha$ en degres', '$C_L$')
    plt.legend(['$\delta _{{PHR}} =  ${:.1f}'.format(ut.deg_of_rad(dm)) for dm in dms], loc='best')
    if filename is not None:
        plt.savefig(filename, dpi=160)
    return figure


def Cm(aircraft, alpha):
    Cma = -aircraft.ms * aircraft.CLa
    return aircraft.Cm0 + Cma * (alpha - aircraft.a0)


def plot_Cm(aircraft, filename=None):
    mss = [-0.1, 0., 0.2, 1.]
    figure = ut.prepare_fig(None, u'Coefficient de moment {}'.format(aircraft.name))
    for ms in mss:
        aircraft.set_mass_and_static_margin(0.5, ms)
        plt.plot(ut.deg_of_rad(alphas), Cm(aircraft, alphas))
    ut.decorate(plt.gca(), u'Coefficient de moment {}'.format(aircraft.name), r'$\alpha$ en degres', '$C_m$',
                ['$ms =  ${: .1f}'.format(ms) for ms in mss])
    if filename is not None:
        plt.savefig(filename, dpi=160)
    return figure


def plot_dphr_e_vt(aircraft, filename=None):
    mss = [-0.1, 0., 0.2, 1.]

    def dphr(alpha, ms):
        return (- aircraft.Cm0 + ms * aircraft.CLa * (alpha - aircraft.a0)) / aircraft.Cmd

    figure = ut.prepare_fig(None, r'$\delta_{phr}$ en fonction de '
                                  r'$\alpha$ avec $V_{t} + 50\%$ {name}'.format(phr='phr', t='t', name=aircraft.name))
    for ms in mss:
        aircraft.set_mass_and_static_margin(aircraft.m_k, ms)

        aircraft.Vt = aircraft.Vt * 1.5
        aircraft.Cmd = -aircraft.Vt * aircraft.CLat
        aircraft.Cmq = aircraft.Cmd * aircraft.CLq

        dphrs = [dphr(alpha, ms) for alpha in alphas]
        plt.plot(ut.deg_of_rad(alphas), dphrs)
        ut.decorate(plt.gca(), r'$\delta_{phr}$ en fonction de $\alpha$, $V_{t} + 50\%$', r'$\alpha$ en degres',
                    r'$\delta_{phr}$', ['$ms =  ${: .1f}'.format(ms) for ms in mss])
    if filename is not None:
        plt.savefig(filename, dpi=160)
    return figure


def plot_dphr_e(aircraft, filename=None):
    mss = [-0.1, 0., 0.2, 1.]

    def dphr(alpha, ms):
        return (- aircraft.Cm0 + ms * aircraft.CLa * (alpha - aircraft.a0)) / aircraft.Cmd

    figure = ut.prepare_fig(None, r'$\delta_{phr}$ en fonction de $\alpha$ {name}'.format(
        phr='phr', t='t', name=aircraft.name))
    for ms in mss:
        aircraft.set_mass_and_static_margin(aircraft.m_k, ms)
        dphrs = [dphr(alpha, ms) for alpha in alphas]
        plt.plot(ut.deg_of_rad(alphas), dphrs)
        ut.decorate(plt.gca(), r'$\delta_{phr}$ en fonction de $\alpha$', r'$\alpha$ en degres', u'$\delta_{phr}$',
                    ['$ms =  ${: .1f}'.format(ms) for ms in mss])
    if filename is not None:
        plt.savefig(filename, dpi=160)
    return figure


def plot_CLe(aircraft, filename=None):
    mss = [0.2, 1.]

    def dphr(alpha, ms):
        return (- aircraft.Cm0 + ms * aircraft.CLa * (alpha - aircraft.a0)) / aircraft.Cmd

    def CLe(alpha, ms):
        return dyn.get_aero_coefs(1, alpha, 0, dphr(alpha, ms), aircraft)[0]

    figure = ut.prepare_fig(None, r'$\delta_{phr}$ en fonction de $\alpha$ {name}'.format(
        phr='phr', t='t', name=aircraft.name))

    for ms in mss:
        aircraft.set_mass_and_static_margin(aircraft.m_k, ms)
        CLes = [CLe(alpha, ms) for alpha in alphas]
        plt.plot(ut.deg_of_rad(alphas), CLes)
        ut.decorate(plt.gca(), r'$C_{Le}$ en fonction de $\alpha$', r'$\alpha$ en degres', u'$C_{Le}$',
                    ['$ms =  ${: .1f}'.format(ms) for ms in mss])

    if filename is not None:
        plt.savefig(filename, dpi=160)
    return figure


def plot_polar(aircraft, filename=None):
    mss = [0.2, 1.]

    def dphr(alpha, ms):
        return (- aircraft.Cm0 + ms * aircraft.CLa * (alpha - aircraft.a0)) / aircraft.Cmd

    def CLe(alpha, ms):
        return dyn.get_aero_coefs(1, alpha, 0, dphr(alpha, ms), aircraft)[0]

    def CDe(alpha, ms):
        return dyn.get_aero_coefs(1, alpha, 0, dphr(alpha, ms), aircraft)[1]

    figure = ut.prepare_fig(None, r'$\Polaires équilibrées {name}'.format(name=aircraft.name))

    finesses = []
    for ms in mss:
        aircraft.set_mass_and_static_margin(aircraft.m_k, ms)
        CLes = [CLe(alpha, ms) for alpha in alphas]
        CDes = [CDe(alpha, ms) for alpha in alphas]

        pentes = np.array(CLes) / np.array(CDes)
        finesse = round(max(pentes), 3)
        finesses.append(finesse)
        # print('finesse : {} pour ms = {}'.format(finesse, ms))

        plt.plot(CDes, CLes)
        ut.decorate(plt.gca(), 'Polaires équilibrées', r'$C_{De}$', u'$C_{Le}$',
                    ['$ms =  ${: .1f}, finesse : {}'.format(ms, finesse) for ms, finesse in zip(mss, finesses)])

    if filename is not None:
        plt.savefig(filename, dpi=160)
    return figure


if __name__ == "__main__":
    aircraft = dyn.Param_A321()
    plot_thrust(aircraft, 'plots/seance_1/{}_thrust.png'.format(aircraft.get_name()))
    plot_CL(aircraft, 'plots/seance_1/{}_CL.png'.format(aircraft.get_name()))
    plot_Cm(aircraft, 'plots/seance_1/{}_Cm.png'.format(aircraft.get_name()))
    plot_dphr_e(aircraft, 'plots/seance_1/{}_dPHR.png'.format(aircraft.get_name()))
    plot_dphr_e_vt(aircraft, 'plots/seance_1/{}_dPHR vt augmenter de 50%.png'.format(aircraft.get_name()))
    plot_CLe(aircraft, 'plots/seance_1/{}_CLe.png'.format(aircraft.get_name()))
    plot_polar(aircraft, 'plots/seance_1/{}_polar.png'.format(aircraft.get_name()))
    plt.show()
